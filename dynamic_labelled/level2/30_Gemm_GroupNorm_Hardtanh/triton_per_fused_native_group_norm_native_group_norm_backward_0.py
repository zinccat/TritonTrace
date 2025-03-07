# From: 30_Gemm_GroupNorm_Hardtanh

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_native_group_norm_native_group_norm_backward_0(
    input_grad_ptr, mean_ptr, inv_std_ptr, input_ptr, grad_output_ptr, mean_grad_ptr, 
    output_grad_ptr, output_ptr, xnumel, rnumel, XBLOCK: tl.constexpr
):
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_block_index = rindex
    x_block_index = xindex
    x_mod_index = xindex % 8
    input_grad = tl.load(input_grad_ptr + (r_block_index + 64 * x_block_index), xmask, other=0.0)
    mean = tl.load(mean_ptr + (x_block_index), xmask)
    inv_std = tl.load(inv_std_ptr + (x_block_index), xmask)
    input_data = tl.load(input_ptr + (r_block_index + 64 * x_mod_index), xmask, eviction_policy='evict_last', other=0.0)
    grad_output = tl.load(grad_output_ptr + (r_block_index + 64 * x_mod_index), xmask, eviction_policy='evict_last', other=0.0)
    mean_grad = tl.load(mean_grad_ptr + (r_block_index + 64 * x_block_index), xmask, other=0.0)
    inv_std_evict = tl.load(inv_std_ptr + (x_block_index), xmask, eviction_policy='evict_last')
    mean_evict = tl.load(mean_ptr + (x_block_index), xmask, eviction_policy='evict_last')
    
    input_grad_centered = input_grad - mean
    scaled_input_grad = input_grad_centered * inv_std
    scaled_grad_output = scaled_input_grad * grad_output
    sum_scaled_grad_output = scaled_grad_output + grad_output
    lower_bound = -2.0
    upper_bound = 2.0
    clamp_mask = (sum_scaled_grad_output <= lower_bound) | (sum_scaled_grad_output >= upper_bound)
    clamped_mean_grad = tl.where(clamp_mask, 0.0, mean_grad)
    clamped_scaled_grad = clamped_mean_grad * input_grad
    scaled_clamped_grad = clamped_scaled_grad * grad_output
    broadcast_scaled_clamped_grad = tl.broadcast_to(scaled_clamped_grad, [XBLOCK, RBLOCK])
    masked_broadcast = tl.where(xmask, broadcast_scaled_clamped_grad, 0)
    sum_masked_broadcast = tl.sum(masked_broadcast, 1)[:, None]
    
    clamped_scaled_grad_broadcast = tl.broadcast_to(clamped_scaled_grad, [XBLOCK, RBLOCK])
    masked_clamped_scaled_grad = tl.where(xmask, clamped_scaled_grad_broadcast, 0)
    sum_masked_clamped_scaled_grad = tl.sum(masked_clamped_scaled_grad, 1)[:, None]
    
    scaled_inv_std = inv_std_evict * grad_output
    clamped_scaled_inv_std = clamped_mean_grad * scaled_inv_std
    sum_mean_evict = sum_masked_clamped_scaled_grad * mean_evict
    mean_diff = sum_mean_evict - sum_masked_broadcast
    mean_diff_scaled = mean_diff * inv_std_evict
    mean_diff_scaled_cubed = mean_diff_scaled * mean_diff_scaled * mean_diff_scaled
    scaling_factor = 0.015625
    scaled_mean_diff = mean_diff_scaled_cubed * scaling_factor
    scaled_input_grad_scaled = input_grad * scaled_mean_diff
    sum_scaled_mean_diff = sum_mean_evict + scaled_input_grad_scaled
    neg_scaled_mean_diff = -scaled_mean_diff
    neg_scaled_mean_diff_scaled = neg_scaled_mean_diff * mean_evict
    mean_diff_scaled_scaled = sum_mean_evict * inv_std_evict
    neg_scaled_mean_diff_scaled_factor = neg_scaled_mean_diff_scaled - (mean_diff_scaled_scaled * scaling_factor)
    final_output = sum_scaled_mean_diff + neg_scaled_mean_diff_scaled_factor
    
    tl.store(output_grad_ptr + (r_block_index + 64 * x_block_index), sum_scaled_grad_output, xmask)
    tl.store(output_ptr + (r_block_index + 64 * x_block_index), final_output, xmask)