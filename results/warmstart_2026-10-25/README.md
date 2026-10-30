# Warm-started RBM runs for the 9x9 skyrmion model

Josh Laurienzo, 2026-10-25

`scripts/scripts_netket/scripts/gs9x9_warmstart.py` is a companion to `gs9x9_validate.py`
with the same lattice, Hamiltonian (Pauli units, oriented x/y bonds), RBM (alpha=2, complex)
and Berg-Luscher post-processing. It differs in three places:

1. the classical unit-vector minimum of the same Hamiltonian on the same 9x9 torus is
   computed first (L-BFGS from random starts) and used as the benchmark `E_cl`;
2. `INIT=warm` writes that texture into the RBM visible biases, so the optimisation starts
   at `E_cl` instead of at the uniform |+x> product state (E = -162);
3. `OPT=sr` uses stochastic reconfiguration (Sgd + SR, holomorphic) instead of plain Adam.

Why: the cold-started Adam runs in this repository plateau at E = -186.45 (run 1002_172802,
120k iterations). The classical minimum of the committed Hamiltonian on the 9x9 torus is
E_cl = -188.31 (a Q = 2 texture), so a variational energy above it cannot be the ground
state. Starting from the uniform state, gradient descent falls into the Q = 0 basin
(classical floor -184.26) and the RBM only dresses that state.

## Runs (one A100 each, 3000 iterations, 8192 samples, chunk 2048, seed 0)

| arm | optimiser | E_final | var | Q | cos(angle to classical texture) | wall |
|---|---|---|---|---|---|---|
| warm_sr | Sgd lr 0.01 + SR diag_shift 0.01 | -190.425 +- 0.011 | 2.0 | 2.000 | 0.998 (min 0.996) | 1.8 h |
| warm_adam | Adam lr 1e-3 | -190.163 +- 0.015 | 3.7 | 2.000 | 0.988 (min 0.960) | 1.6 h |

Both arms pass the convention check printed at the start of `run.log`: with kernel and
hidden biases at zero the RBM reproduces E_cl to sampling error and every site's <sigma>
to within 0.025. Per directory: `run.log` (stdout), `*.log` (NetKet JsonLog), `*.mpack`
(final weights), `*_spin_profile.npz` (per-site <sigma>, plus the classical seed texture),
`*_classical_texture.npy`, `*_summary.json`.

Open points: one seed per arm; the classical Q = 1 state lies only 0.07 above Q = 2, so the
sector of the quantum ground state is not settled (seed from the Q = 1 texture to compare);
NetKet warns that 8192 samples is fewer than the 13365 parameters and recommends
`nk.driver.VMC_SR` for production.
