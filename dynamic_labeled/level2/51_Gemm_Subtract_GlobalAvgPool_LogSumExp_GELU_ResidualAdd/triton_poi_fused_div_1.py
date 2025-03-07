# From: 51_Gemm_Subtract_GlobalAvgPool_LogSumExp_GELU_ResidualAdd

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_div_1(input_ptr0, input_ptr1, output_ptr0, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    block_id = index // 512
    local_index = index

    input_val0 = tl.load(input_ptr0 + (block_id), mask, eviction_policy='evict_last')
    input_val1 = tl.load(input_ptr1 + (block_id), mask, eviction_policy='evict_last')

    abs_input_val1 = tl.math.abs(input_val1)
    inf_value = float("inf")
    is_inf = abs_input_val1 == inf_value
    zero_value = 0.0
    safe_input_val1 = tl.where(is_inf, zero_value, input_val1)

    adjusted_input_val1 = input_val1 - safe_input_val1
    exp_adjusted = tl.math.exp(adjusted_input_val1)
    log_exp = tl.math.log(exp_adjusted)
    log_sum_exp = log_exp + safe_input_val1

    sqrt_2_over_2 = 0.7071067811865476
    erf_input = log_sum_exp * sqrt_2_over_2
    erf_result = tl.extra.cuda.libdevice.erf(erf_input)

    one_value = 1.0
    erf_plus_one = erf_result + one_value
    half_value = 0.5
    erf_half = erf_plus_one * half_value

    squared_log_sum_exp = log_sum_exp * log_sum_exp
    neg_half = -0.5
    exp_neg_half_squared = tl.math.exp(squared_log_sum_exp * neg_half)

    sqrt_2_pi = 0.3989422804014327
    gaussian = exp_neg_half_squared * sqrt_2_pi
    gaussian_scaled = log_sum_exp * gaussian

    gelu_result = erf_half + gaussian_scaled
    scaled_input_val0 = input_val0 * gelu_result

    subtract_log_sum_exp = input_val1 - log_sum_exp
    exp_subtract = tl.math.exp(subtract_log_sum_exp)
    final_result = scaled_input_val0 * exp_subtract

    scale_factor = 0.001953125
    scaled_final_result = final_result * scale_factor

    tl.store(output_ptr0 + (local_index), scaled_final_result, mask)