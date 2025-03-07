# From: 93_ConvTranspose2d_Add_Min_GELU_Multiply

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused_add_convolution_gelu_lift_fresh_minimum_mul_0(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    # Calculate indices
    x3 = x_index
    x1 = (x_index // 4356) % 16
    
    # Load data
    input_output_value = tl.load(in_out_ptr0 + (x3), None)
    input_value = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    
    # Perform addition
    added_value = input_output_value + input_value
    
    # Constants
    half = 0.5
    zero = 0.0
    sqrt_half = 0.7071067811865476
    one = 1.0
    two = 2.0
    
    # Calculate minimum
    min_value = triton_helpers.minimum(added_value + half, zero)
    
    # Intermediate calculations
    min_half_product = min_value * half
    min_sqrt_half_product = min_value * sqrt_half
    erf_result = tl.extra.cuda.libdevice.erf(min_sqrt_half_product)
    erf_plus_one = erf_result + one
    gelu_result = min_half_product * erf_plus_one * two
    
    # Store results
    tl.store(in_out_ptr0 + (x3), added_value, None)
    tl.store(out_ptr0 + (x3), gelu_result, None)