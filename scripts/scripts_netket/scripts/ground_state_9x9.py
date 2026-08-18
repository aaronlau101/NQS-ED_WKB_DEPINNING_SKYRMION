import os
# Must be set before importing JAX/NetKet
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["JAX_ENABLE_X64"] = "True"

import numpy as np
import jax
import jax.numpy as jnp
import optax
import netket as nk

print("1. Setting up 9x9 Production Lattice")
L = 9
# Using Square with color_edges=True to easily distinguish x and y bonds
graph = nk.graph.Square(length=L, pbc=True, color_edges=True)
hi = nk.hilbert.Spin(s=0.5, N=graph.n_nodes)

print("2. Building Hamiltonian Operators")
# Parameters in units of J
J = 1.0
D = 1.0     # DMI strength
A = 0.5     # Easy-axis anisotropy
Bz = 1.0    # Magnetic field

ha = nk.operator.LocalOperator(hi, dtype=complex)

# Pauli Matrices
X, Y, Z = nk.operator.spin.sigmax, nk.operator.spin.sigmay, nk.operator.spin.sigmaz

# filter_color=0: bonds along x-axis. filter_color=1: bonds along y-axis
x_edges = graph.edges(filter_color=0)
y_edges = graph.edges(filter_color=1)

# X-Axis Bonds
for u, v in x_edges:
    # Heisenberg Exchange
    ha += J * (X(hi, u) @ X(hi, v) + Y(hi, u) @ Y(hi, v) + Z(hi, u) @ Z(hi, v))
    
    # Bloch DMI along x
    ha += D * (Y(hi, u) @ Z(hi, v) - Z(hi, u) @ Y(hi, v))

# Y-Axis Bonds
for u, v in y_edges:
    # Heisenberg Exchange
    ha += J * (X(hi, u) @ X(hi, v) + Y(hi, u) @ Y(hi, v) + Z(hi, u) @ Z(hi, v))
    
    # Bloch DMI along y
    ha += D * (Z(hi, u) @ X(hi, v) - X(hi, u) @ Z(hi, v))

# Loop over sites for On-site Anisotropy and Zeeman field
for u in graph.nodes():
    pass
# Easy-axis Anisotropy
for u, v in graph.edges():
    ha += -A * (Z(hi, u) @ Z(hi, v))
# Zeeman Field (Single site terms)
for u in graph.nodes():
    ha += -Bz * Z(hi, u)

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
sampler = nk.sampler.MetropolisLocal(hi)
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
# Creates 'ground_state_9x9_results.json' and 'ground_state_9x9_results.wft'
logger = nk.logging.JsonLog("ground_state_9x9_results", save_params_every=1000)

# Fixed: Removed illegal 'log_interval' argument. 
# Added 'show_progress=True' to output progress bars directly to the terminal stdout.
driver.run(
    n_iter=120000, 
    out=logger,
    show_progress=True
)

print("Optimization Complete. Outputs and weights saved to ground_state_9x9_results.*")