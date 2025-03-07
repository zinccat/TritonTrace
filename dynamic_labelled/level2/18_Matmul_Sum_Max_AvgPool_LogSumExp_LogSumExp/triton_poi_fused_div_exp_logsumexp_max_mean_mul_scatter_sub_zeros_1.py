# From: 18_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_div_exp_logsumexp_max_mean_mul_scatter_sub_zeros_1poi_fused_div_exp_logsumexp_max_mean_mul_scatter_sub_zeros_1(
    input_ptr0, input_ptr1, input_ptr2, output_ptr0, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    base_index = index

    input_data0 = tl.load(input_ptr0 + (base_index), mask)
    input_data1 = tl.load(input_ptr1 + (base_index), mask)
    input_data2 = tl.load(input_ptr2 + (base_index), mask)

    divisor = 1.0
    division_result = input_data1 / divisor
    absolute_value = tl.math.abs(division_result)
    infinity = float("inf")
    is_infinity = absolute_value == infinity
    zero_value = 0.0
    adjusted_division_result = tl.where(is_infinity, zero_value, division_result)

    adjusted_value = division_result - adjusted_division_result
    exp_result = tl.math.exp(adjusted_value)
    log_result = tl.math.log(exp_result)
    logsumexp_result = log_result + adjusted_division_result

    difference = logsumexp_result - input_data2
    exp_difference = tl.math.exp(difference)

    multiplied_result = input_data0 * exp_difference
    exp_adjusted_difference = division_result - logsumexp_result
    exp_exp_adjusted_difference = tl.math.exp(exp_adjusted_difference)

    final_result = multiplied_result * exp_exp_adjusted_difference * divisor
    tl.store(output_ptr0 + (base_index), final_result, mask)