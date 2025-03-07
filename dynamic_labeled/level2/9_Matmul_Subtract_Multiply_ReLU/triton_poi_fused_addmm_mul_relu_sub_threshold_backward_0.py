# From: 9_Matmul_Subtract_Multiply_ReLU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_mul_relu_sub_threshold_backward_0(
    in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 5)
    
    # Load data from input pointers
    input_data = tl.load(in_out_ptr0 + (x2), xmask)
    weight_data = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    
    # Perform addition
    added_result = input_data + weight_data
    
    # Subtract constant
    subtracted_result = added_result - 2.0
    
    # Multiply by constant
    multiplied_result = subtracted_result * 1.5
    
    # Apply ReLU (max with zero)
    relu_result = triton_helpers.maximum(tl.full([1], 0, tl.int32), multiplied_result)
    
    # Compare with threshold
    threshold = 0.0
    is_below_threshold = relu_result <= threshold
    
    # Store results
    tl.store(in_out_ptr0 + (x2), relu_result, xmask)
    tl.store(out_ptr0 + (x2), is_below_threshold, xmask)