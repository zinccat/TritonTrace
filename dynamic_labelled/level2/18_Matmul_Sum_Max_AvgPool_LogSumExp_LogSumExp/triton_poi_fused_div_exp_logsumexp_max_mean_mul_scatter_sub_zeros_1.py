# From: 18_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_div_exp_logsumexp_max_mean_mul_scatter_sub_zeros_1(
    input_ptr0, input_ptr1, input_ptr2, output_ptr0, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    indices = offset + tl.arange(0, XBLOCK)[:]
    mask = indices < num_elements
    base_indices = indices

    input_values0 = tl.load(input_ptr0 + (base_indices), mask)
    input_values1 = tl.load(input_ptr1 + (base_indices), mask)
    input_values2 = tl.load(input_ptr2 + (base_indices), mask)

    divisor = 1.0
    division_result = input_values1 / divisor
    absolute_values = tl.math.abs(division_result)
    infinity = float("inf")
    is_infinity = absolute_values == infinity
    zero_value = 0.0
    adjusted_values = tl.where(is_infinity, zero_value, division_result)

    adjusted_subtraction = division_result - adjusted_values
    exponentiated_values = tl.math.exp(adjusted_subtraction)
    log_values = tl.math.log(exponentiated_values)
    log_adjusted_values = log_values + adjusted_values

    difference = log_adjusted_values - input_values2
    exp_difference = tl.math.exp(difference)

    multiplied_values = input_values0 * exp_difference
    subtraction_result = division_result - log_adjusted_values
    exp_subtraction = tl.math.exp(subtraction_result)

    final_result = multiplied_values * exp_subtraction * divisor
    tl.store(output_ptr0 + (base_indices), final_result, mask)