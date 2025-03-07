# From: 75_Gemm_GroupNorm_Min_BiasAdd

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_native_group_norm_backward_3(
    input_grad_ptr, input_ptr, input_mean_ptr, input_var_ptr, input_norm_ptr, output_grad_ptr, 
    xnumel, rnumel, XBLOCK: tl.constexpr
):
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    row_index = rindex
    col_index = xindex
    col_sub_index = xindex % 8
    grad_input = tl.load(input_grad_ptr + (row_index + 32 * col_index), xmask, other=0.0)
    grad_output = tl.load(input_ptr + (row_index + 32 * col_index), xmask, other=0.0)
    mean = tl.load(input_mean_ptr + (row_index + 32 * col_sub_index), xmask, eviction_policy='evict_last', other=0.0)
    variance = tl.load(input_var_ptr + (col_index), xmask, eviction_policy='evict_last')
    norm_factor = tl.load(input_norm_ptr + (col_index), xmask, eviction_policy='evict_last')
    
    grad_input_times_grad_output = grad_input * grad_output
    grad_input_times_grad_output_times_mean = grad_input_times_grad_output * mean
    broadcast_grad_input_times_grad_output_times_mean = tl.broadcast_to(grad_input_times_grad_output_times_mean, [XBLOCK, RBLOCK])
    masked_grad_input_times_grad_output_times_mean = tl.where(xmask, broadcast_grad_input_times_grad_output_times_mean, 0)
    sum_grad_input_times_grad_output_times_mean = tl.sum(masked_grad_input_times_grad_output_times_mean, 1)[:, None]
    
    grad_input_times_mean = grad_input * mean
    broadcast_grad_input_times_mean = tl.broadcast_to(grad_input_times_mean, [XBLOCK, RBLOCK])
    masked_grad_input_times_mean = tl.where(xmask, broadcast_grad_input_times_mean, 0)
    sum_grad_input_times_mean = tl.sum(masked_grad_input_times_mean, 1)[:, None]
    
    norm_factor_times_mean = norm_factor * mean
    grad_input_times_norm_factor_times_mean = grad_input * norm_factor_times_mean
    sum_grad_input_times_mean_times_norm_factor = sum_grad_input_times_mean * norm_factor
    sum_grad_input_times_mean_times_norm_factor_minus_sum_grad_input_times_grad_output_times_mean = sum_grad_input_times_mean_times_norm_factor - sum_grad_input_times_grad_output_times_mean
    
    sum_grad_input_times_mean_times_norm_factor_minus_sum_grad_input_times_grad_output_times_mean_times_norm_factor_times_mean = sum_grad_input_times_mean_times_norm_factor_minus_sum_grad_input_times_grad_output_times_mean * norm_factor_times_mean
    sum_grad_input_times_mean_times_norm_factor_minus_sum_grad_input_times_grad_output_times_mean_times_norm_factor_times_mean_times_norm_factor_times_mean = sum_grad_input_times_mean_times_norm_factor_minus_sum_grad_input_times_grad_output_times_mean_times_norm_factor_times_mean * norm_factor_times_mean
    sum_grad_input_times_mean_times_norm_factor_minus_sum_grad_input_times_grad_output_times_mean_times_norm_factor_times_mean_times_norm_factor_times_mean_times_norm_factor_times_mean = sum_grad_input_times_mean_times_norm_factor_minus_sum_grad_input_times_grad_output_times_mean_times_norm_factor_times_mean_times_norm_factor_times_mean * norm_factor_times_mean
    
    scale_factor = 0.03125
    scaled_variance = sum_grad_input_times_mean_times_norm_factor_minus_sum_grad_input_times_grad_output_times_mean_times_norm_factor_times_mean_times_norm_factor_times_mean_times_norm_factor_times_mean * scale_factor
    grad_output_times_scaled_variance = grad_output * scaled_variance
    grad_input_times_norm_factor_times_mean_plus_grad_output_times_scaled_variance = grad_input_times_norm_factor_times_mean + grad_output_times_scaled_variance
    
    negative_scaled_variance = -scaled_variance
    negative_scaled_variance_times_norm_factor = negative_scaled_variance * norm_factor
    sum_grad_input_times_mean_times_norm_factor_minus_sum_grad_input_times_grad_output_times_mean_times_norm_factor = sum_grad_input_times_mean_times_norm_factor_minus_sum_grad_input_times_grad_output_times_mean * norm_factor
    sum_grad_input_times_mean_times_norm_factor_minus_sum_grad_input_times_grad_output_times_mean_times_norm_factor_times_scale_factor = sum_grad_input_times_mean_times_norm_factor_minus_sum_grad_input_times_grad_output_times_mean_times_norm_factor * scale_factor
    negative_scaled_variance_times_norm_factor_minus_sum_grad_input_times_mean_times_norm_factor_minus_sum_grad_input_times_grad_output_times_mean_times_norm_factor_times_scale_factor = negative_scaled_variance_times_norm_factor - sum_grad_input_times_mean_times_norm_factor_minus_sum_grad_input_times_grad_output_times_mean_times_norm_factor_times_scale_factor
    
    final_output = grad_input_times_norm_factor_times_mean_plus_grad_output_times_scaled_variance + negative_scaled_variance_times_norm_factor_minus_sum_grad_input_times_mean_times_norm_factor_minus_sum_grad_input_times_grad_output_times_mean_times_norm_factor_times_scale_factor
    
    tl.store(output_grad_ptr + (row_index + 32 * col_index), final_output, xmask)