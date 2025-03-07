# From: 40_Matmul_Scaling_ResidualAdd

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_addmm_mul_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 128)
    
    # Load data from input pointers
    loaded_in_out = tl.load(in_out_ptr0 + (x2), xmask)
    loaded_in = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    
    # Perform addition
    added_result = loaded_in_out + loaded_in
    
    # Scale and add
    scale_factor = 0.5
    scaled_result = added_result * scale_factor
    final_result = scaled_result + added_result
    
    # Store the result back
    tl.store(in_out_ptr0 + (x2), final_result, xmask)