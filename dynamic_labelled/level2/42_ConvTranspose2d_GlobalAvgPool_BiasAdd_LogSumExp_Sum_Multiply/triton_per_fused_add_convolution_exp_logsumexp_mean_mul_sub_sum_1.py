# From: 42_ConvTranspose2d_GlobalAvgPool_BiasAdd_LogSumExp_Sum_Multiply

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_convolution_exp_logsumexp_mean_mul_sub_sum_1per_fused_add_convolution_exp_logsumexp_mean_mul_sub_sum_1(
    in_out_ptr0, in_ptr0, out_ptr2, kernel_size, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr
):
    RBLOCK: tl.constexpr = 16
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < input_num_elements
    r_index = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = r_index
    x0 = x_index
    input_data = tl.load(in_out_ptr0 + (r1 + 16 * x0), x_mask, other=0.0)
    kernel_data = tl.load(in_ptr0 + (r1), None, eviction_policy='evict_last')
    kernel_elements = 4 + kernel_size * kernel_size + 4 * kernel_size
    kernel_elements_float = kernel_elements.to(tl.float32)
    normalized_input = input_data / kernel_elements_float
    combined_data = normalized_input + kernel_data
    broadcasted_data = tl.broadcast_to(combined_data, [XBLOCK, RBLOCK])
    masked_data = tl.where(x_mask, broadcasted_data, float("-inf"))
    max_values = triton_helpers.max2(masked_data, 1)[:, None]
    abs_max_values = tl.math.abs(max_values)
    inf_value = float("inf")
    is_inf = abs_max_values == inf_value
    safe_max_values = tl.where(is_inf, 0.0, max_values)
    shifted_data = combined_data - safe_max_values
    exp_shifted_data = tl.math.exp(shifted_data)
    broadcasted_exp_data = tl.broadcast_to(exp_shifted_data, [XBLOCK, RBLOCK])
    masked_exp_data = tl.where(x_mask, broadcasted_exp_data, 0)
    sum_exp_data = tl.sum(masked_exp_data, 1)[:, None]
    log_sum_exp = tl.math.log(sum_exp_data)
    log_sum_exp_adjusted = log_sum_exp + safe_max_values
    final_exp_data = combined_data - log_sum_exp_adjusted
    exp_final_data = tl.math.exp(final_exp_data)
    scale_factor = 10.0
    scaled_log_sum_exp = log_sum_exp_adjusted * scale_factor
    tl.store(in_out_ptr0 + (r1 + 16 * x0), exp_final_data, x_mask)
    tl.store(out_ptr2 + (x0), scaled_log_sum_exp, x_mask)