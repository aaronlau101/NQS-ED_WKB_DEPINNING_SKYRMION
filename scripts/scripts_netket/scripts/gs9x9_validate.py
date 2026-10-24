import os
# Must be set before importing JAX/NetKet
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["JAX_ENABLE_X64"] = "True"

import numpy as np
import jax
import jax.numpy as jnp
import optax
import netket as nk
import datetime

print("1. Setting up 9x9 Production Lattice")
L = 9
# Using Square with color_edges=True to easily distinguish x and y bonds
graph = nk.graph.Square(length=L, pbc=True, color_edges=True)
hi = nk.hilbert.Spin(s=0.5, N=graph.n_nodes)

print("2. Building Hamiltonian Operators")
# Parameters in units of J
J = 1.0
D = 1.0     # DMI strength
A = 0.5     # Heisenberg anisotropy
Bz = 1.0    # Magnetic field

ha = nk.operator.LocalOperator(hi, dtype=complex)

# Pauli Matrices
X, Y, Z = nk.operator.spin.sigmax, nk.operator.spin.sigmay, nk.operator.spin.sigmaz

pos = graph.positions

def orient(u, v, axis):
    d = pos[v, axis] - pos[u, axis]
    d = (d + L / 2) % L - L / 2          # minimum-image displacement, gives ±1
    return (u, v) if d > 0 else (v, u)

# Enforce that every edge tuple (u, v) strictly points in the +x or +y direction
x_edges = [orient(u, v, 0) for u, v in graph.edges(filter_color=0)]
y_edges = [orient(u, v, 1) for u, v in graph.edges(filter_color=1)]

# x-directed nearest-neighbor bonds
# u_ij = x_hat
# H_exchange = -J (sigma_x sigma_x + sigma_y sigma_y)
# H_anisotropy = -A sigma_z sigma_z
# -D (u_ij x z_hat) . (sigma_i x sigma_j)
# = D (sigma_i^z sigma_j^x - sigma_i^x sigma_j^z)

# X-Axis Bonds
for u, v in x_edges:
    # XY Heisenberg exchange
    ha += -J * (
        X(hi, u) @ X(hi, v)
        + Y(hi, u) @ Y(hi, v)
    )
    # Heisenberg anisotropy
    ha += -A * (
        Z(hi, u) @ Z(hi, v)
    )
    # DMI for u_ij = +x_hat
    ha += D * (
        Z(hi, u) @ X(hi, v)
        - X(hi, u) @ Z(hi, v)
    )

# y-directed nearest-neighbor bonds
# u_ij = y_hat
# -D (u_ij x z_hat) . (sigma_i x sigma_j)
# = -D (sigma_i^y sigma_j^z - sigma_i^z sigma_j^y)
# = D (sigma_i^z sigma_j^y - sigma_i^y sigma_j^z)

# Y-Axis Bonds
for u, v in y_edges:

    # XY Heisenberg exchange
    ha += -J * (
        X(hi, u) @ X(hi, v)
        + Y(hi, u) @ Y(hi, v)
    )

    # Heisenberg anisotropy
    ha += -A * (
        Z(hi, u) @ Z(hi, v)
    )

    # DMI for u_ij = +y_hat
    ha += D * (
        Z(hi, u) @ Y(hi, v)
        - Y(hi, u) @ Z(hi, v)
    )

# Loop over sites for Zeeman field (single site terms)
# H_field = +B^z sum_i sigma_i^z
for u in graph.nodes():
    ha += Bz * Z(hi, u)

ha = ha.to_jax_operator()   # connected configs generated on-the-fly per chunk on GPU

print("3. Initializing RBM Ansatz (alpha=2)")
def complex_normal(key, shape, dtype=jnp.complex128):
    key1, key2 = jax.random.split(key)
    cr = jax.nn.initializers.normal(stddev=0.01)(key1, shape, jnp.float64)
    ci = jax.nn.initializers.normal(stddev=0.01)(key2, shape, jnp.float64)
    return cr + 1j * ci

ma = nk.models.RBM(
    alpha=2, 
    param_dtype=complex,
    kernel_init=complex_normal,
    hidden_bias_init=complex_normal,
    visible_bias_init=complex_normal
)

print("4. Configuring Scheduled Optimizer and Sampler")
sampler = nk.sampler.MetropolisLocal(hi, sweep_size=graph.n_nodes)
n_samples_energy = 2**14
# Pass chunk_size to break up sample into parts
vstate = nk.vqs.MCState(
    sampler, 
    ma, 
    n_samples=n_samples_energy, 
    chunk_size=2**11
)

print("5. Optimizer with Piecewise Decay Schedule")
# FIX: Swapped to piecewise_constant to match factors of 10 drops exactly at boundaries
lr_schedule = optax.piecewise_constant_schedule(
    init_value=1e-3,
    boundaries_and_scales={
        40000: 0.1,   # Drops to 1e-4
        80000: 0.1    # Drops to 1e-5
    }
)

op = optax.adam(learning_rate=lr_schedule, b1=0.9, b2=0.999)
driver = nk.VMC(ha, op, variational_state=vstate)

print("6. Starting Ground State Optimization")
# Generate a unique timestamp prefix (YearMonthDay_HourMinuteSecond)
timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
log_filename = f"ground_state_9x9_results_{timestamp}"

# Creates 'ground_state_9x9_MMDD_HHMMSS.json' and '.wft'
logger = nk.logging.JsonLog(log_filename, save_params_every=1000)

# Fixed: Removed illegal 'log_interval' argument. 
# Added 'show_progress=True' to output progress bars directly to the terminal stdout.
driver.run(
    n_iter=120000, 
    out=logger,
    show_progress=True
)

print(f"Optimization Complete. Outputs and weights saved to {log_filename}.*")

# Measure expectation values of sigma_x, sigma_y, sigma_z at each site.
sx_ops = [X(hi, i) for i in range(graph.n_nodes)]
sy_ops = [Y(hi, i) for i in range(graph.n_nodes)]
sz_ops = [Z(hi, i) for i in range(graph.n_nodes)]

mx_vals = np.array(
    [vstate.expect(op).mean.real for op in sx_ops],
    dtype=float
)
my_vals = np.array(
    [vstate.expect(op).mean.real for op in sy_ops],
    dtype=float
)
mz_vals = np.array(
    [vstate.expect(op).mean.real for op in sz_ops],
    dtype=float
)

# Explicitly map each site using graph.positions.
spin_profile = np.zeros((L, L, 3), dtype=float)
site_seen = np.zeros((L, L), dtype=bool)

for idx, p in enumerate(pos):
    x, y = map(int, p[:2])

    if not (0 <= x < L and 0 <= y < L):
        raise ValueError(
            f"Unexpected lattice coordinate ({x}, {y}) for site {idx}."
        )

    if site_seen[x, y]:
        raise ValueError(
            f"Duplicate lattice coordinate ({x}, {y})."
        )

    spin_profile[x, y] = [
        mx_vals[idx],
        my_vals[idx],
        mz_vals[idx]
    ]
    site_seen[x, y] = True

if not np.all(site_seen):
    raise ValueError("Some lattice coordinates are missing.")

# Check local expectation-vector magnitudes before normalization.
norm_m = np.linalg.norm(spin_profile, axis=-1)

print(f"Minimum local spin-vector norm: {norm_m.min():.8e}")
print(f"Maximum local spin-vector norm: {norm_m.max():.8e}")
print(
    "Number of near-zero local vectors:",
    int(np.sum(norm_m < 1e-10))
)

# Normalization is undefined for zero or near-zero vectors.
if np.any(norm_m < 1e-10):
    raise ValueError(
        "Cannot reliably compute the normalized spin texture: "
        "at least one local expectation vector has near-zero magnitude. "
        "Inspect the spin profile before computing the topological charge."
    )

m_unit = spin_profile / norm_m[..., None]

# Save raw and normalized spin profiles, including coordinates.
np.savez(
    f"spin_profile_{timestamp}.npz",
    x=pos[:, 0],
    y=pos[:, 1],
    raw=spin_profile,
    unit=m_unit,
    mx_site=mx_vals,
    my_site=my_vals,
    mz_site=mz_vals
)

# Also save a plain-text spin profile for convenient offline analysis.
x_grid, y_grid = np.meshgrid(
    np.arange(L), np.arange(L), indexing="ij"
)

np.savetxt(
    f"spin_profile_{timestamp}.txt",
    np.column_stack((
        x_grid.ravel(),
        y_grid.ravel(),
        spin_profile.reshape(-1, 3)
    )),
    header="x y S_x S_y S_z",
    fmt="%.10f"
)

print("8. Computing Berg-Luscher Lattice Topological Charge")

def solid_angle(s1, s2, s3):
    # The inputs are normalized spin vectors.
    numerator = np.dot(s1, np.cross(s2, s3))
    denominator = (
        1.0
        + np.dot(s1, s2)
        + np.dot(s2, s3)
        + np.dot(s3, s1)
    )

    return 2.0 * np.arctan2(numerator, denominator)


Q_total = 0.0

# Periodic boundary conditions.
# Split each square into two oriented triangles.
for x in range(L):
    for y in range(L):
        p00 = m_unit[x, y]
        p10 = m_unit[(x + 1) % L, y]
        p01 = m_unit[x, (y + 1) % L]
        p11 = m_unit[(x + 1) % L, (y + 1) % L]

        Q_total += solid_angle(p00, p10, p01)
        Q_total += solid_angle(p11, p01, p10)

Q_skyrmion = Q_total / (4.0 * np.pi)

print(f"Berg-Luscher Skyrmion Number: Q = {Q_skyrmion:.8f}")
print(f"Nearest Integer: {round(Q_skyrmion)}")

# Save the result and relevant diagnostic information.
with open(f"skyrmion_number_{timestamp}.txt", "w") as f:
    f.write(f"Q_skyrmion = {Q_skyrmion:.12f}\n")
    f.write(f"Nearest integer = {round(Q_skyrmion)}\n")
    f.write(f"Lattice size = {L}x{L}\n")
    f.write("Method = Berg-Luscher solid-angle triangulation\n")
    f.write("Boundary conditions = periodic\n")

print("Post-processing complete.")
print(f"Spin profiles saved to spin_profile_{timestamp}.npz and .txt")
print(f"Skyrmion number saved to skyrmion_number_{timestamp}.txt")