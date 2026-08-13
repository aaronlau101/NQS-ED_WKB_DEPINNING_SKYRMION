import os
# Force calculation to run on GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Import libraries
import numpy as np
import jax
import jax.numpy as jnp
import optax
import netket as nk

print("1. Initializing Lattice Graph and Hilbert Space...")
# 3x3 square grid with Periodic Boundary Conditions
graph = nk.graph.Grid(extent=[3, 3], pbc=True)
hilbert = nk.hilbert.Spin(s=1/2, N=graph.n_nodes)

print("2. Constructing Local Hamiltonian Operators...")
J, D, Bz = 1.0, 1.0, 1.0

# Define explicitly mapped Pauli matrices
sigma_x = 0.5 * np.array([[0, 1], [1, 0]], dtype=np.complex128)
sigma_y = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
sigma_z = 0.5 * np.array([[1, 0], [0, -1]], dtype=np.complex128)

H = nk.operator.LocalOperator(hilbert, dtype=np.complex128)

for edge in graph.edges():
    i, j = edge[0], edge[1]
    
    # 1. Heisenberg Exchange: J * (SxSx + SySy + SzSz)
    S_xx = np.kron(sigma_x, sigma_x)
    S_yy = np.kron(sigma_y, sigma_y)
    S_zz = np.kron(sigma_z, sigma_z)
    H += J * nk.operator.LocalOperator(hilbert, S_xx + S_yy + S_zz, [i, j])
    
    # 2. DMI term: D * (SxSy - SySx)
    S_xy = np.kron(sigma_x, sigma_y)
    S_yx = np.kron(sigma_y, sigma_x)
    H += D * nk.operator.LocalOperator(hilbert, S_xy - S_yx, [i, j])

for site in range(graph.n_nodes):
    H -= Bz * nk.operator.LocalOperator(hilbert, sigma_z, [site])

print("3. Building RBM Ansatz with Complex Parameters...")
machine = nk.models.RBM(
    alpha=2, 
    param_dtype=np.complex128,
    kernel_init=jax.nn.initializers.normal(stddev=0.01)
)

print("4. Instantiating MCMC Sampler and Optimizer...")
sampler = nk.sampler.MetropolisLocal(hilbert)
optimizer = optax.adam(learning_rate=1e-3)

variational_state = nk.vqs.MCState(sampler, machine, n_samples=2016)
vmc = nk.VMC(H, optimizer, variational_state=variational_state)

print("5. Running optimization steps...")
for step in range(200):
    vmc.run(n_iter=1, show_progress=False)
    
    # Correct way to get the energy expectation value of the current state
    energy = vmc.state.expect(H)
    
    if step % 20 == 0:
        print(f"Step {step:3d} | Variational Energy: {energy.mean.real:.6f} (Variance: {energy.variance.real:.4f})")

print("\n--- Final VMC Validation Result ---")
print(f"Target Ground Energy (ED): -4.811737")
print(f"Achieved VMC Energy:       {vmc.state.expect(H).mean.real:.6f}")