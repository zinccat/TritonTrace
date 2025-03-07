# From: 18_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_logsumexp_max_mean_sum_0poi_fused_logsumexp_max_mean_sum_0(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex

    # Load input data with eviction policy
    input0 = tl.load(in_ptr0 + (5 * x0), xmask, eviction_policy='evict_last')
    input1 = tl.load(in_ptr0 + (1 + 5 * x0), xmask, eviction_policy='evict_last')
    input2 = tl.load(in_ptr0 + (2 + 5 * x0), xmask, eviction_policy='evict_last')
    input3 = tl.load(in_ptr0 + (3 + 5 * x0), xmask, eviction_policy='evict_last')
    input4 = tl.load(in_ptr0 + (4 + 5 * x0), xmask, eviction_policy='evict_last')

    # Compute sum
    sum_result = input0 + input1 + input2 + input3 + input4

    # Compute mean
    mean_result = sum_result / 1.0

    # Compute max
    abs_mean = tl.math.abs(mean_result)
    max_value = float("inf")
    is_max = abs_mean == max_value
    max_result = tl.where(is_max, 0.0, mean_result)

    # Compute logsumexp
    adjusted_mean = mean_result - max_result
    exp_adjusted_mean = tl.math.exp(adjusted_mean)
    log_exp_adjusted_mean = tl.math.log(exp_adjusted_mean)
    logsumexp_result = log_exp_adjusted_mean + max_result

    # Compute max of logsumexp
    abs_logsumexp = tl.math.abs(logsumexp_result)
    is_max_logsumexp = abs_logsumexp == max_value
    max_logsumexp_result = tl.where(is_max_logsumexp, 0.0, logsumexp_result)

    # Final logsumexp adjustment
    adjusted_logsumexp = logsumexp_result - max_logsumexp_result
    exp_adjusted_logsumexp = tl.math.exp(adjusted_logsumexp)
    final_logsumexp = tl.math.log(exp_adjusted_logsumexp) + max_logsumexp_result

    # Store results
    tl.store(out_ptr0 + (x0), sum_result, xmask)
    tl.store(out_ptr1 + (x0), final_logsumexp, xmask)