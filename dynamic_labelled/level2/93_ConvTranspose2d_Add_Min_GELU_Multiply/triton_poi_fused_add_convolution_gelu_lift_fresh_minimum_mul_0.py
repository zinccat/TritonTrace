# From: 93_ConvTranspose2d_Add_Min_GELU_Multiply

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_convolution_gelu_lift_fresh_minimum_mul_0poi_fused_add_convolution_gelu_lift_fresh_minimum_mul_0(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4356) % 16)
    
    # Load data from input pointers
    input_data = tl.load(in_out_ptr0 + (x3), xmask)
    additional_data = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    
    # Perform addition
    added_data = input_data + additional_data
    
    # Constants for GELU calculation
    half = 0.5
    zero = 0.0
    sqrt_two_over_pi = 0.7071067811865476
    one = 1.0
    two = 2.0
    
    # Calculate minimum
    min_value = triton_helpers.minimum(added_data + half, zero)
    
    # Intermediate GELU calculations
    scaled_min = min_value * half
    erf_input = min_value * sqrt_two_over_pi
    erf_result = tl.extra.cuda.libdevice.erf(erf_input)
    gelu_intermediate = scaled_min * (erf_result + one)
    
    # Final GELU result
    gelu_result = gelu_intermediate * two
    
    # Store results
    tl.store(in_out_ptr0 + (x3), added_data, xmask)
    tl.store(out_ptr0 + (x3), gelu_result, xmask)