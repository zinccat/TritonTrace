# From: 18_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_logsumexp_max_mean_sum_0(input_ptr, output_sum_ptr, output_logsumexp_ptr, num_elements, XBLOCK: tl.constexpr):
    num_elements = 128
    offset = tl.program_id(0) * XBLOCK
    indices = offset + tl.arange(0, XBLOCK)[:]
    mask = indices < num_elements
    base_indices = indices

    # Load elements from input
    element_0 = tl.load(input_ptr + (5 * base_indices), mask, eviction_policy='evict_last')
    element_1 = tl.load(input_ptr + (1 + (5 * base_indices)), mask, eviction_policy='evict_last')
    element_2 = tl.load(input_ptr + (2 + (5 * base_indices)), mask, eviction_policy='evict_last')
    element_3 = tl.load(input_ptr + (3 + (5 * base_indices)), mask, eviction_policy='evict_last')
    element_4 = tl.load(input_ptr + (4 + (5 * base_indices)), mask, eviction_policy='evict_last')

    # Compute sum
    sum_01 = element_0 + element_1
    sum_012 = sum_01 + element_2
    sum_0123 = sum_012 + element_3
    sum_01234 = sum_0123 + element_4

    # Compute mean
    divisor = 1.0
    mean_value = sum_01234 / divisor

    # Compute max
    abs_mean = tl.math.abs(mean_value)
    max_value = float("inf")
    is_max_inf = abs_mean == max_value
    max_value_replaced = tl.where(is_max_inf, 0.0, mean_value)

    # Compute logsumexp
    shifted_mean = mean_value - max_value_replaced
    exp_shifted_mean = tl.math.exp(shifted_mean)
    log_exp_shifted_mean = tl.math.log(exp_shifted_mean)
    logsumexp_value = log_exp_shifted_mean + max_value_replaced

    abs_logsumexp = tl.math.abs(logsumexp_value)
    is_logsumexp_inf = abs_logsumexp == max_value
    logsumexp_value_replaced = tl.where(is_logsumexp_inf, 0.0, logsumexp_value)

    # Final logsumexp computation
    shifted_logsumexp = logsumexp_value - logsumexp_value_replaced
    exp_shifted_logsumexp = tl.math.exp(shifted_logsumexp)
    final_logsumexp = tl.math.log(exp_shifted_logsumexp) + logsumexp_value_replaced

    # Store results
    tl.store(output_sum_ptr + (base_indices), sum_01234, mask)
    tl.store(output_logsumexp_ptr + (base_indices), final_logsumexp, mask)