import os
os.environ["JAX_ENABLE_X64"] = "True"

import netket as nk
import flax
import jax.numpy as jnp

print("1. Reconstructing 9x9 Production Lattice & Hilbert Space")
L = 9
graph = nk.graph.Square(length=L, pbc=True, color_edges=True)
hi = nk.hilbert.Spin(s=0.5, N=graph.n_nodes)

print("2. Reconstructing RBM Ansatz (alpha=2)")
ma = nk.models.RBM(alpha=2, param_dtype=complex)

print("3. Instantiating a Template Variational State")
sampler = nk.sampler.MetropolisLocal(hi)
vstate = nk.vqs.MCState(sampler, ma, n_samples=16) 

mpack_filename = "ground_state_9x9_results.mpack" 

print(f"4. Loading weights from {mpack_filename}...")
if os.path.exists(mpack_filename):
    with open(mpack_filename, "rb") as file:
        vstate.variables = flax.serialization.from_bytes(vstate.variables, file.read())
    print("✓ Weights loaded successfully!")
    
    print("\n Model Parameters Overview")
    top_params = vstate.variables['params']
    
    # 1. Extract visible_bias from the top level
    visible_bias = top_params['visible_bias']
    
    # 2. Extract kernel and hidden_bias from the nested 'Dense' sub-layer
    dense_layer = top_params['Dense']
    kernel = dense_layer['kernel']
    hidden_bias = dense_layer['bias']  # FIX: Flax names this 'bias' inside Dense layers
    
    # Safe printing of shapes
    print(f"Kernel (weights) shape:       {kernel.shape}")
    print(f"Visible bias shape:           {visible_bias.shape}")
    print(f"Hidden bias shape:            {hidden_bias.shape}")
    
    # Example: Print a small subset of the kernel weights to confirm they loaded cleanly
    print("\nSample kernel weights (top-left 3x3 block):")
    print(kernel[:3, :3])

    
else:
    print(f"✗ Error: Could not find file '{mpack_filename}'. Check your path.")
