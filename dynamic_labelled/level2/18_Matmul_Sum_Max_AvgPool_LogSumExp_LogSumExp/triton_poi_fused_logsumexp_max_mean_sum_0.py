# From: 18_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_logsumexp_max_mean_sum_0(input_ptr, output_sum_ptr, output_logsumexp_ptr, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = indices < num_elements
    base_indices = indices

    # Load elements with eviction policy
    element_0 = tl.load(input_ptr + (5 * base_indices), mask, eviction_policy='evict_last')
    element_1 = tl.load(input_ptr + (1 + 5 * base_indices), mask, eviction_policy='evict_last')
    element_2 = tl.load(input_ptr + (2 + 5 * base_indices), mask, eviction_policy='evict_last')
    element_3 = tl.load(input_ptr + (3 + 5 * base_indices), mask, eviction_policy='evict_last')
    element_4 = tl.load(input_ptr + (4 + 5 * base_indices), mask, eviction_policy='evict_last')

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
    inf_value = float("inf")
    is_inf = abs_mean == inf_value
    max_value = tl.where(is_inf, 0.0, mean_value)

    # Compute logsumexp
    adjusted_mean = mean_value - max_value
    exp_adjusted_mean = tl.math.exp(adjusted_mean)
    log_exp_adjusted_mean = tl.math.log(exp_adjusted_mean)
    logsumexp_value = log_exp_adjusted_mean + max_value

    # Compute max of logsumexp
    abs_logsumexp = tl.math.abs(logsumexp_value)
    is_inf_logsumexp = abs_logsumexp == inf_value
    max_logsumexp = tl.where(is_inf_logsumexp, 0.0, logsumexp_value)

    # Final logsumexp adjustment
    adjusted_logsumexp = logsumexp_value - max_logsumexp
    exp_adjusted_logsumexp = tl.math.exp(adjusted_logsumexp)
    log_exp_adjusted_logsumexp = tl.math.log(exp_adjusted_logsumexp)
    final_logsumexp = log_exp_adjusted_logsumexp + max_logsumexp

    # Store results
    tl.store(output_sum_ptr + (base_indices), sum_01234, mask)
    tl.store(output_logsumexp_ptr + (base_indices), final_logsumexp, mask)