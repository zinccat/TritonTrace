# From: 93_ConvTranspose2d_Add_Min_GELU_Multiply

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_div_eq_gelu_gelu_backward_gt_lift_fresh_masked_fill_minimum_mul_where_0poi_fused_add_div_eq_gelu_gelu_backward_gt_lift_fresh_masked_fill_minimum_mul_where_0(
    in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex

    # Load input data
    input_data = tl.load(in_out_ptr0 + (x0), xmask)
    input_tensor = tl.load(in_ptr0 + (x0), xmask)

    # Constants
    half = 0.5
    zero = 0.0
    two = 2.0
    sqrt_half = 0.7071067811865476
    one = 1.0
    erf_coeff = 0.3989422804014327
    neg_half = -0.5

    # Operations
    input_data_plus_half = input_data + half
    greater_than_zero = input_data_plus_half > zero
    equal_to_zero = input_data_plus_half == zero

    input_tensor_times_two = input_tensor * two
    min_input_data_zero = triton_helpers.minimum(input_data_plus_half, zero)
    min_input_data_zero_times_sqrt_half = min_input_data_zero * sqrt_half

    erf_result = tl.extra.cuda.libdevice.erf(min_input_data_zero_times_sqrt_half)
    erf_result_plus_one = erf_result + one
    erf_result_plus_one_times_half = erf_result_plus_one * half

    min_input_data_zero_squared = min_input_data_zero * min_input_data_zero
    exp_result = tl.math.exp(min_input_data_zero_squared * neg_half)
    exp_result_times_erf_coeff = exp_result * erf_coeff
    exp_result_times_erf_coeff_times_min_input_data_zero = exp_result_times_erf_coeff * min_input_data_zero

    gelu_result = erf_result_plus_one_times_half + exp_result_times_erf_coeff_times_min_input_data_zero
    input_tensor_times_gelu_result = input_tensor_times_two * gelu_result
    input_tensor_times_gelu_result_times_half = input_tensor_times_gelu_result * half

    # Conditional operations
    where_equal_to_zero = tl.where(equal_to_zero, input_tensor_times_gelu_result_times_half, input_tensor_times_gelu_result)
    where_greater_than_zero = tl.where(greater_than_zero, zero, where_equal_to_zero)

    # Store result
    tl.store(in_out_ptr0 + (x0), where_greater_than_zero, xmask)