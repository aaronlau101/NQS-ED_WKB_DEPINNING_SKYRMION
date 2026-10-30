"""gs9x9_warmstart.py -- 9x9 skyrmion-model RBM VMC with a classical warm start and SR.

Companion to gs9x9_validate.py: identical lattice, Hamiltonian (Pauli units, oriented
x/y bonds), RBM, and Berg-Luscher post-processing. Differences:
  1. the classical (unit-vector) minimum of the SAME Hamiltonian on the SAME torus is
     computed first (L-BFGS, many starts) and used as the benchmark E_cl;
  2. INIT=warm encodes that texture in the RBM visible biases, so the run starts at
     E_cl instead of the uniform |+x> product state (-162);
  3. OPT=sr uses stochastic reconfiguration (natural gradient) instead of plain Adam.
Knobs (environment):
  INIT=warm|cold   OPT=sr|adam   N_ITER (3000)   LR (0.01 sr / 1e-3 adam)   DIAG_SHIFT (0.01)
  N_SAMPLES (8192)   CHUNK (2048)   SEED (0)   TAG   OUT_DIR (.)   N_CL_STARTS (40)
Pick the GPU in the shell (e.g. CUDA_VISIBLE_DEVICES=1 python gs9x9_warmstart.py); the script does not pin one.
"""
import os, sys, json, time, datetime
os.environ.setdefault("JAX_ENABLE_X64", "True")
import numpy as np
from scipy.optimize import minimize
import jax, jax.numpy as jnp, optax, netket as nk, flax

INIT = os.environ.get("INIT", "warm"); OPT = os.environ.get("OPT", "sr")
N_ITER = int(os.environ.get("N_ITER", 3000))
LR = float(os.environ.get("LR", 0.01 if OPT == "sr" else 1e-3))
DIAG_SHIFT = float(os.environ.get("DIAG_SHIFT", 0.01))
N_SAMPLES = int(os.environ.get("N_SAMPLES", 2**13)); CHUNK = int(os.environ.get("CHUNK", 2**11))
SEED = int(os.environ.get("SEED", 0)); N_CL_STARTS = int(os.environ.get("N_CL_STARTS", 40))
TAG = os.environ.get("TAG", f"{INIT}_{OPT}")
OUT_DIR = os.environ.get("OUT_DIR", "."); os.makedirs(OUT_DIR, exist_ok=True)
stamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
prefix = os.path.join(OUT_DIR, f"gs9x9_{TAG}_{stamp}")
T_START = time.time()
print(f"[cfg] INIT={INIT} OPT={OPT} N_ITER={N_ITER} LR={LR} DIAG_SHIFT={DIAG_SHIFT} "
      f"N_SAMPLES={N_SAMPLES} CHUNK={CHUNK} SEED={SEED} TAG={TAG}")
print(f"[env] netket {nk.__version__} jax {jax.__version__} devices {jax.devices()}")
if not any(d.platform == "gpu" for d in jax.devices()):
    print("FATAL: no GPU backend visible to jax; refusing CPU fallback"); sys.exit(4)

# ------------------------------------------------------------------ 1. lattice + H (as in gs9x9_validate.py)
L = 9
graph = nk.graph.Square(length=L, pbc=True, color_edges=True)
hi = nk.hilbert.Spin(s=0.5, N=graph.n_nodes)
N = graph.n_nodes
J, D, A, Bz = 1.0, 1.0, 0.5, 1.0
X, Y, Z = nk.operator.spin.sigmax, nk.operator.spin.sigmay, nk.operator.spin.sigmaz
pos = graph.positions

def orient(u, v, axis):
    d = pos[v, axis] - pos[u, axis]
    d = (d + L / 2) % L - L / 2
    return (u, v) if d > 0 else (v, u)

x_edges = [orient(u, v, 0) for u, v in graph.edges(filter_color=0)]
y_edges = [orient(u, v, 1) for u, v in graph.edges(filter_color=1)]
assert len(x_edges) == N and len(y_edges) == N

ha = nk.operator.LocalOperator(hi, dtype=complex)
for u, v in x_edges:
    ha += -J * (X(hi, u) @ X(hi, v) + Y(hi, u) @ Y(hi, v))
    ha += -A * (Z(hi, u) @ Z(hi, v))
    ha += D * (Z(hi, u) @ X(hi, v) - X(hi, u) @ Z(hi, v))
for u, v in y_edges:
    ha += -J * (X(hi, u) @ X(hi, v) + Y(hi, u) @ Y(hi, v))
    ha += -A * (Z(hi, u) @ Z(hi, v))
    ha += D * (Z(hi, u) @ Y(hi, v) - Y(hi, u) @ Z(hi, v))
for u in graph.nodes():
    ha += Bz * Z(hi, u)
ha_jax = ha.to_jax_operator()

# ------------------------------------------------------------------ 2. classical minimum on the SAME bonds
def classical_bonds():
    Mex = np.diag([-J, -J, -A])
    Dx = np.zeros((3, 3)); Dx[2, 0] = D; Dx[0, 2] = -D    # n_u.(Dx n_v) = D(nz_u nx_v - nx_u nz_v)
    Dy = np.zeros((3, 3)); Dy[2, 1] = D; Dy[1, 2] = -D
    us, vs, Ms = [], [], []
    for u, v in x_edges: us.append(u); vs.append(v); Ms.append(Mex + Dx)
    for u, v in y_edges: us.append(u); vs.append(v); Ms.append(Mex + Dy)
    return np.array(us), np.array(vs), np.array(Ms)
BONDS = classical_bonds()

def cl_energy_grad(n):
    u, v, M = BONDS
    Mv = np.einsum('bij,bj->bi', M, n[v])
    E = np.einsum('bi,bi->', n[u], Mv) + Bz * n[:, 2].sum()
    g = np.zeros_like(n); np.add.at(g, u, Mv); np.add.at(g, v, np.einsum('bji,bj->bi', M, n[u])); g[:, 2] += Bz
    return E, g

def cl_fg(m_flat):
    m = m_flat.reshape(N, 3); r = np.linalg.norm(m, axis=1, keepdims=True); n = m / r
    E, g = cl_energy_grad(n)
    return E, ((g - (g * n).sum(1, keepdims=True) * n) / r).ravel()

def cl_relax(n0):
    res = minimize(cl_fg, n0.ravel().copy(), jac=True, method='L-BFGS-B',
                   options=dict(maxiter=20000, ftol=1e-15, gtol=1e-10, maxcor=30))
    m = res.x.reshape(N, 3); n = m / np.linalg.norm(m, axis=1, keepdims=True)
    return cl_energy_grad(n)[0], n

def site_xy(i):
    return int(round(pos[i, 0])), int(round(pos[i, 1]))
XY = np.array([site_xy(i) for i in range(N)]); IDX = {tuple(xy): i for i, xy in enumerate(XY)}

def berg_luscher(n):  # n: (N,3) unit vectors indexed by site
    def sa(s1, s2, s3):
        return 2 * np.arctan2(np.dot(s1, np.cross(s2, s3)), 1 + s1 @ s2 + s2 @ s3 + s3 @ s1)
    Q = 0.0
    for x in range(L):
        for y in range(L):
            p00 = n[IDX[(x, y)]]; p10 = n[IDX[((x + 1) % L, y)]]
            p01 = n[IDX[(x, (y + 1) % L)]]; p11 = n[IDX[((x + 1) % L, (y + 1) % L)]]
            Q += sa(p00, p10, p01) + sa(p11, p01, p10)
    return Q / (4 * np.pi)

t0 = time.time(); rng = np.random.default_rng(12345)
best = (np.inf, None)
for k in range(N_CL_STARTS):
    E, n = cl_relax(rng.normal(size=(N, 3)))
    if E < best[0] - 1e-9: best = (E, n)
E_cl, n_cl = best; Q_cl = berg_luscher(n_cl)
print(f"[classical] min over {N_CL_STARTS} random starts: E_cl={E_cl:.6f} (E/N={E_cl/N:.6f}) "
      f"Q={Q_cl:.4f} <nz>={n_cl[:,2].mean():.4f}  ({time.time()-t0:.1f}s)")
print(f"[classical] references: |+x> product={-2.0*N:.2f}  canted FM={-2.25*N:.2f}")
np.save(prefix + "_classical_texture.npy", n_cl)

# ------------------------------------------------------------------ 3. RBM + warm start
def complex_normal(key, shape, dtype=jnp.complex128):
    k1, k2 = jax.random.split(key)
    return (jax.nn.initializers.normal(stddev=0.01)(k1, shape, jnp.float64)
            + 1j * jax.nn.initializers.normal(stddev=0.01)(k2, shape, jnp.float64))

ma = nk.models.RBM(alpha=2, param_dtype=complex, kernel_init=complex_normal,
                   hidden_bias_init=complex_normal, visible_bias_init=complex_normal)
sampler = nk.sampler.MetropolisLocal(hi, sweep_size=N)
vstate = nk.vqs.MCState(sampler, ma, n_samples=N_SAMPLES, chunk_size=CHUNK, n_discard_per_chain=32,
                        seed=SEED, sampler_seed=SEED)

# which local-state VALUE is spin-up (sigma_z = +1)?  sigmaz is diag(+1,-1) in local-state index order
sz1 = np.real(np.diag(nk.operator.spin.sigmaz(nk.hilbert.Spin(0.5, N=1), 0).to_dense()))
ls = np.array([float(v) for v in hi.local_states])
s_up = float(ls[int(np.argmax(sz1))]); s_dn = float(ls[int(np.argmin(sz1))])
print(f"[conv] local_states={ls.tolist()} sigmaz diag={sz1.tolist()} -> up value {s_up:+.0f}, down value {s_dn:+.0f}")

def product_state_visible_bias(n, theta_min=0.02):
    theta = np.arccos(np.clip(n[:, 2], -1, 1)); theta = np.clip(theta, theta_min, np.pi - theta_min)
    phi = np.arctan2(n[:, 1], n[:, 0])
    log_ratio = np.log(1 / np.tan(theta / 2)) - 1j * phi        # psi(up)/psi(down) = cot(th/2) e^{-i phi}
    return log_ratio / (s_up - s_dn)                            # log psi = sum_i a_i s_i

params = flax.core.unfreeze(vstate.parameters)
E_cold = float(vstate.expect(ha_jax).mean.real)
print(f"[init] cold RBM (tiny random weights ~ |+x>^N): E={E_cold:.4f}")
if INIT == "warm":
    a = product_state_visible_bias(n_cl)
    params["visible_bias"] = jnp.asarray(a)
    k0 = jax.random.PRNGKey(SEED)
    params["Dense"]["kernel"] = jnp.zeros_like(params["Dense"]["kernel"])
    params["Dense"]["bias"] = jnp.zeros_like(params["Dense"]["bias"])
    vstate.parameters = params
    # --- convention check on the pure product state (W=b=0): energy and a few local moments
    est = vstate.expect(ha_jax); E_prod = float(est.mean.real)
    print(f"[warm] pure product state: E={E_prod:.4f} +- {float(est.error_of_mean):.4f}  (classical {E_cl:.4f})")
    mchk = np.array([[float(vstate.expect(O(hi, i)).mean.real) for O in (X, Y, Z)] for i in range(N)])
    dev = np.abs(mchk - n_cl)
    print(f"[warm] |<sigma> - n_classical| over all sites: max per component (x,y,z)={np.round(dev.max(0),4).tolist()}, mean={dev.mean():.4f}")
    for i in [0, 10, 40, 80]:
        print(f"       site {i:2d}: <sigma>={np.round(mchk[i],3).tolist()}  classical n={np.round(n_cl[i],3).tolist()}")
    if dev.max() > 0.08 or abs(E_prod - E_cl) > 1.0:
        print("FATAL: warm-start convention mismatch (sign/ordering); refusing to run"); sys.exit(5)
    # --- break hidden-unit symmetry with the usual small noise, keep the texture
    params["Dense"]["kernel"] = complex_normal(jax.random.fold_in(k0, 1), params["Dense"]["kernel"].shape)
    params["Dense"]["bias"] = complex_normal(jax.random.fold_in(k0, 2), params["Dense"]["bias"].shape)
    vstate.parameters = params
E_init = float(vstate.expect(ha_jax).mean.real)
print(f"[init] starting point for optimisation: E={E_init:.4f}")

# ------------------------------------------------------------------ 4. optimiser
if OPT == "sr":
    op = nk.optimizer.Sgd(learning_rate=LR)
    sr = nk.optimizer.SR(diag_shift=DIAG_SHIFT, holomorphic=True)
    try:
        driver = nk.VMC(ha_jax, op, variational_state=vstate, preconditioner=sr)
    except TypeError as exc:   # newer NetKet without the preconditioner kwarg
        print("[opt] nk.VMC(preconditioner=) unavailable:", exc, "-> using nk.driver.VMC_SR")
        driver = nk.driver.VMC_SR(ha_jax, op, variational_state=vstate, diag_shift=DIAG_SHIFT, use_ntk=False)
else:
    op = optax.adam(learning_rate=LR, b1=0.9, b2=0.999)
    driver = nk.VMC(ha_jax, op, variational_state=vstate)

logger = nk.logging.JsonLog(prefix, save_params_every=100)
def cb(step, log_data, driver):
    if step % 100 == 0 or step == N_ITER - 1:
        e = log_data["Energy"]
        print(f"[it {step:6d}] E={float(e.mean.real):11.5f}  E/N={float(e.mean.real)/N:9.5f}  "
              f"var={float(e.variance):8.4f}  Rhat={float(e.R_hat):6.3f}  "
              f"acc={float(driver.state.sampler_state.acceptance):.3f}  t={time.time()-T_START:8.1f}s", flush=True)
    return True
driver.run(n_iter=N_ITER, out=logger, callback=cb, show_progress=False)

# ------------------------------------------------------------------ 5. post-processing (as in gs9x9_validate.py)
vstate.n_samples = 2**14
est = vstate.expect(ha_jax)
E_fin = float(est.mean.real); err_fin = float(est.error_of_mean); var_fin = float(est.variance)
print(f"[final] E={E_fin:.5f} +- {err_fin:.5f}  var={var_fin:.4f}  E/N={E_fin/N:.5f}  "
      f"vs classical {E_cl:.5f}  (E - E_cl = {E_fin - E_cl:+.4f})")
m = np.zeros((N, 3))
for i in range(N):
    m[i] = [float(vstate.expect(O(hi, i)).mean.real) for O in (X, Y, Z)]
norm = np.linalg.norm(m, axis=1)
print(f"[final] local |<sigma>|: min={norm.min():.4f} max={norm.max():.4f} mean={norm.mean():.4f}")
m_unit = m / np.maximum(norm, 1e-12)[:, None]
Q = berg_luscher(m_unit)
ovl = (m_unit * n_cl).sum(1)
print(f"[final] Berg-Luscher Q={Q:.5f} (classical texture Q={Q_cl:.4f}); "
      f"cos(angle to classical texture): mean={ovl.mean():.3f} min={ovl.min():.3f}")
spin_profile = np.zeros((L, L, 3)); 
for i in range(N):
    x, y = XY[i]; spin_profile[x, y] = m[i]
np.savez(prefix + "_spin_profile.npz", x=pos[:, 0], y=pos[:, 1], raw=spin_profile,
         unit=spin_profile / np.maximum(np.linalg.norm(spin_profile, axis=-1), 1e-12)[..., None],
         mx_site=m[:, 0], my_site=m[:, 1], mz_site=m[:, 2], classical_texture=n_cl)
print("[final] <sigma_z> map (rows=y, cols=x):")
for y in range(L):
    print("   " + " ".join(f"{spin_profile[x, y, 2]:6.2f}" for x in range(L)))
summary = dict(tag=TAG, init=INIT, opt=OPT, n_iter=N_ITER, lr=LR, diag_shift=DIAG_SHIFT, n_samples=N_SAMPLES,
               seed=SEED, E_classical=E_cl, Q_classical=Q_cl, E_cold_init=E_cold, E_init=E_init,
               E_final=E_fin, E_final_err=err_fin, var_final=var_fin, E_minus_Ecl=E_fin - E_cl,
               Q_final=Q, cos_to_classical_mean=float(ovl.mean()), cos_to_classical_min=float(ovl.min()),
               local_moment_min=float(norm.min()), local_moment_mean=float(norm.mean()),
               wall_s=time.time() - T_START, netket=nk.__version__, log_prefix=prefix)
json.dump(summary, open(prefix + "_summary.json", "w"), indent=1)
print("[done]", json.dumps(summary))
