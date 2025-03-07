# From: 26_ConvTranspose3d_Add_HardSwish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_hardswish_hardswish_backward_mul_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    
    input_value = tl.load(in_ptr0 + (x0), None)
    output_value = tl.load(in_out_ptr0 + (x0), None)
    
    add_constant = 3.0
    max_value = 6.0
    min_value = 0.0
    scale_factor = 0.16666666666666666
    negative_threshold = -3.0
    scale_factor_2 = 0.3333333333333333
    offset = 0.5
    
    intermediate_sum = output_value + add_constant
    clamped_value = triton_helpers.maximum(intermediate_sum, min_value)
    bounded_value = triton_helpers.minimum(clamped_value, max_value)
    
    scaled_output = output_value * bounded_value
    scaled_result = scaled_output * scale_factor
    
    product_input_output = input_value * output_value
    
    scaled_output_2 = output_value * scale_factor_2
    adjusted_value = scaled_output_2 + offset
    conditional_product = product_input_output * adjusted_value
    
    conditional_result = tl.where(output_value <= add_constant, conditional_product, product_input_output)
    final_conditional_result = tl.where(output_value < negative_threshold, min_value, conditional_result)
    
    final_result = scaled_result + final_conditional_result
    
    tl.store(in_out_ptr0 + (x0), final_result, None)