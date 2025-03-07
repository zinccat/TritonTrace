# From: 42_ConvTranspose2d_GlobalAvgPool_BiasAdd_LogSumExp_Sum_Multiply

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_convolution_exp_logsumexp_mean_mul_sub_sum_1(
    in_out_ptr0, in_ptr0, out_ptr2, kernel_size, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr
):
    RBLOCK: tl.constexpr = 16
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < input_num_elements
    r_indices = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_indices_adjusted = r_indices
    x_indices_adjusted = x_indices
    loaded_values = tl.load(in_out_ptr0 + (r_indices_adjusted + 16 * x_indices_adjusted), x_mask, other=0.0)
    kernel_values = tl.load(in_ptr0 + (r_indices_adjusted), None, eviction_policy='evict_last')
    kernel_size_adjusted = 4 + kernel_size * kernel_size + 4 * kernel_size
    kernel_size_float = kernel_size_adjusted.to(tl.float32)
    normalized_values = loaded_values / kernel_size_float
    combined_values = normalized_values + kernel_values
    broadcasted_values = tl.broadcast_to(combined_values, [XBLOCK, RBLOCK])
    masked_values = tl.where(x_mask, broadcasted_values, float("-inf"))
    max_values = triton_helpers.max2(masked_values, 1)[:, None]
    abs_max_values = tl.math.abs(max_values)
    inf_value = float("inf")
    is_inf = abs_max_values == inf_value
    safe_max_values = tl.where(is_inf, 0.0, max_values)
    adjusted_values = combined_values - safe_max_values
    exp_values = tl.math.exp(adjusted_values)
    broadcasted_exp_values = tl.broadcast_to(exp_values, [XBLOCK, RBLOCK])
    masked_exp_values = tl.where(x_mask, broadcasted_exp_values, 0)
    sum_exp_values = tl.sum(masked_exp_values, 1)[:, None]
    log_sum_exp_values = tl.math.log(sum_exp_values)
    log_sum_exp_adjusted = log_sum_exp_values + safe_max_values
    final_exp_values = combined_values - log_sum_exp_adjusted
    final_values = tl.math.exp(final_exp_values)
    scale_factor = 10.0
    scaled_log_sum_exp = log_sum_exp_adjusted * scale_factor
    tl.store(in_out_ptr0 + (r_indices_adjusted + 16 * x_indices_adjusted), final_values, x_mask)
    tl.store(out_ptr2 + (x_indices_adjusted), scaled_log_sum_exp, x_mask)