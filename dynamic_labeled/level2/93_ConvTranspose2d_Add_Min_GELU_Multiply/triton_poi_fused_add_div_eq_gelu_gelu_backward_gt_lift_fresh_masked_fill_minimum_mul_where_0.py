# From: 93_ConvTranspose2d_Add_Min_GELU_Multiply

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_div_eq_gelu_gelu_backward_gt_lift_fresh_masked_fill_minimum_mul_where_0(
    in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex

    # Load input data
    input_data = tl.load(in_out_ptr0 + (x0), xmask)
    additional_data = tl.load(in_ptr0 + (x0), xmask)

    # Constants
    half = 0.5
    zero = 0.0
    two = 2.0
    sqrt_half = 0.7071067811865476
    one = 1.0
    erf_coeff = 0.3989422804014327
    neg_half = -0.5

    # Operations
    input_plus_half = input_data + half
    is_greater_than_zero = input_plus_half > zero
    is_equal_to_zero = input_plus_half == zero

    scaled_additional_data = additional_data * two
    min_input_zero = triton_helpers.minimum(input_plus_half, zero)
    scaled_min = min_input_zero * sqrt_half

    erf_result = tl.extra.cuda.libdevice.erf(scaled_min)
    erf_plus_one = erf_result + one
    erf_half = erf_plus_one * half

    squared_min = min_input_zero * min_input_zero
    exp_component = tl.math.exp(squared_min * neg_half)
    exp_coeff = exp_component * erf_coeff
    exp_term = min_input_zero * exp_coeff

    gelu_result = erf_half + exp_term
    scaled_gelu = scaled_additional_data * gelu_result * half

    # Conditional selection
    where_equal_to_zero = tl.where(is_equal_to_zero, scaled_gelu, scaled_additional_data * gelu_result)
    where_greater_than_zero = tl.where(is_greater_than_zero, zero, where_equal_to_zero)

    # Store result
    tl.store(in_out_ptr0 + (x0), where_greater_than_zero, xmask)