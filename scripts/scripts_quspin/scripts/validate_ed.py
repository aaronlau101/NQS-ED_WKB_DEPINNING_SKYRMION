print("1. Importing libraries")
import numpy as np
import os

# Force single-core execution directly in python to prevent thread lockups
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from quspin.operators import hamiltonian
from quspin.basis import spin_basis_general

print("2. Defining geometry and bonds")
L_x, L_y = 3, 3
N_spins = L_x * L_y

# Defining 2D square lattice with periodic boundary conditions
def get_neighbors(Lx, Ly):
    bonds_x = []
    bonds_y = []
    for y in range(Ly):
        for x in range(Lx):
            i = y * Lx + x
            next_x = y * Lx + ((x + 1) % Lx)
            next_y = ((y + 1) % Ly) * Lx + x
            bonds_x.append((i, next_x))
            bonds_y.append((i, next_y))
    return bonds_x, bonds_y

bx, by = get_neighbors(L_x, L_y)

J, D, Bz = 1.0, 1.0, 1.0
J_eff = J / 4.0
D_eff = D / 4.0
Bz_eff = Bz / 2.0

heisenberg_list = []
dmi_list_1 = []
dmi_list_2 = []
field_list = [[-Bz_eff, i] for i in range(N_spins)]

for i, j in bx + by:
    heisenberg_list.extend([[J_eff, i, j], [J_eff, i, j], [J_eff, i, j]])
    dmi_list_1.append([D_eff, i, j])
    dmi_list_2.append([-D_eff, j, i])

static = [
    ["xx", heisenberg_list[0::3]],
    ["yy", heisenberg_list[1::3]],
    ["zz", heisenberg_list[2::3]],
    ["xy", dmi_list_1],
    ["yx", dmi_list_2],
    ["z", field_list]
]

print("3. Creating spin_basis_general...")
basis = spin_basis_general(N_spins)

print("4. Constructing hamiltonian matrix...")
H = hamiltonian(static, [], basis=basis, dtype=np.complex128)

print("5. Solving ground state energy (Lanczos)...")
E_ground = H.eigsh(k=1, which='SA', return_eigenvectors=False)

print(f"--- QuSpin Validation Result ---")
print(f"Lattice Size: {L_x}x{L_y} ({N_spins} spins)")
print(f"Exact Ground State Energy: {E_ground[0]:.8f}")