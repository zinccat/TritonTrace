# From: 12_Gemm_Multiply_LeakyReLU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_leaky_relu_mul_0(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 512)
    
    # Load data from input pointers
    input_data = tl.load(in_out_ptr0 + (x2), xmask)
    weight_data = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    
    # Perform element-wise addition
    added_result = input_data + weight_data
    
    # Scale the result
    scale_factor = 2.0
    scaled_result = added_result * scale_factor
    
    # Apply Leaky ReLU
    zero_threshold = 0.0
    leaky_relu_slope = 0.1
    is_positive = scaled_result > zero_threshold
    leaky_relu_result = tl.where(is_positive, scaled_result, scaled_result * leaky_relu_slope)
    
    # Store results
    tl.store(out_ptr0 + (x2), is_positive, xmask)
    tl.store(in_out_ptr0 + (x2), leaky_relu_result, xmask)