# From: 42_ConvTranspose2d_GlobalAvgPool_BiasAdd_LogSumExp_Sum_Multiply

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_per_fused_add_convolution_exp_logsumexp_mean_mul_sub_sum_1(
    input_ptr0, input_ptr1, output_ptr2, output_ptr3, xnumel, rnumel, XBLOCK: tl.constexpr
):
    xnumel = 128
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    row_indices = rindex
    col_indices = xindex
    input_data = tl.load(input_ptr0 + (row_indices + (16 * col_indices)), xmask, other=0.0)
    bias_data = tl.load(input_ptr1 + (row_indices), None, eviction_policy='evict_last')
    divisor = 1156.0
    normalized_data = input_data / divisor
    biased_data = normalized_data + bias_data
    broadcasted_data = tl.broadcast_to(biased_data, [XBLOCK, RBLOCK])
    masked_data = tl.where(xmask, broadcasted_data, float("-inf"))
    max_values = triton_helpers.max2(masked_data, 1)[:, None]
    abs_max_values = tl.math.abs(max_values)
    inf_value = float("inf")
    is_inf = abs_max_values == inf_value
    zero_value = 0.0
    safe_max_values = tl.where(is_inf, zero_value, max_values)
    adjusted_data = biased_data - safe_max_values
    exp_data = tl.math.exp(adjusted_data)
    broadcasted_exp_data = tl.broadcast_to(exp_data, [XBLOCK, RBLOCK])
    masked_exp_data = tl.where(xmask, broadcasted_exp_data, 0)
    sum_exp_data = tl.sum(masked_exp_data, 1)[:, None]
    log_sum_exp_data = tl.math.log(sum_exp_data)
    final_adjusted_data = log_sum_exp_data + safe_max_values
    exp_final_adjusted_data = tl.math.exp(biased_data - final_adjusted_data)
    scale_factor = 10.0
    scaled_log_sum_exp_data = final_adjusted_data * scale_factor
    tl.store(output_ptr2 + (row_indices + (16 * col_indices)), exp_final_adjusted_data, xmask)
    tl.store(output_ptr3 + (col_indices), scaled_log_sum_exp_data, xmask)