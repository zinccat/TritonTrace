# From: 26_ConvTranspose3d_Add_HardSwish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_hardswish_hardswish_backward_mul_0poi_fused_add_hardswish_hardswish_backward_mul_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    input_value = tl.load(in_ptr0 + (x0), None)
    output_value = tl.load(in_out_ptr0 + (x0), None)
    add_constant = 3.0
    sum_value = output_value + add_constant
    zero = 0.0
    max_value = triton_helpers.maximum(sum_value, zero)
    upper_bound = 6.0
    clamped_value = triton_helpers.minimum(max_value, upper_bound)
    scaled_output = output_value * clamped_value
    scale_factor = 0.16666666666666666
    scaled_scaled_output = scaled_output * scale_factor
    product_input_output = input_value * scaled_scaled_output
    lower_bound = -3.0
    is_less_than_lower_bound = output_value < lower_bound
    is_within_bounds = output_value <= add_constant
    product_input_output_value = input_value * output_value
    scale_factor_2 = 0.3333333333333333
    scaled_output_value = output_value * scale_factor_2
    half = 0.5
    adjusted_scaled_output = scaled_output_value + half
    product_adjusted_scaled_output = product_input_output_value * adjusted_scaled_output
    conditional_product = tl.where(is_within_bounds, product_adjusted_scaled_output, product_input_output_value)
    final_conditional_product = tl.where(is_less_than_lower_bound, zero, conditional_product)
    result = product_input_output + final_conditional_product
    tl.store(in_out_ptr0 + (x0), result, None)