# From: 30_Gemm_GroupNorm_Hardtanh

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_native_group_norm_native_group_norm_backward_0(
    input_grad_ptr, input_ptr, mean_ptr, variance_ptr, weight_grad_ptr, weight_ptr, output_grad_ptr, output_ptr, 
    xnumel, rnumel, XBLOCK: tl.constexpr
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
    input = tl.load(input_ptr + (x_block_index), xmask)
    mean = tl.load(mean_ptr + (x_block_index), xmask)
    variance = tl.load(variance_ptr + (x_block_index), xmask)
    weight_grad = tl.load(weight_grad_ptr + (r_block_index + 64 * x_mod_index), xmask, eviction_policy='evict_last', other=0.0)
    weight = tl.load(weight_ptr + (r_block_index + 64 * x_mod_index), xmask, eviction_policy='evict_last', other=0.0)
    weight_grad_full = tl.load(input_grad_ptr + (r_block_index + 64 * x_block_index), xmask, other=0.0)
    variance_evict = tl.load(variance_ptr + (x_block_index), xmask, eviction_policy='evict_last')
    input_evict = tl.load(input_ptr + (x_block_index), xmask, eviction_policy='evict_last')

    normalized_input = input_grad - input
    scaled_input = normalized_input * mean
    weighted_input = scaled_input * weight
    weighted_sum = weighted_input + weight_grad
    lower_bound = -2.0
    upper_bound = 2.0
    clamp_mask = (weighted_sum <= lower_bound) | (weighted_sum >= upper_bound)
    clamped_weight_grad = tl.where(clamp_mask, 0.0, weight_grad_full)
    clamped_weighted_input = clamped_weight_grad * input_grad
    scaled_clamped_weighted_input = clamped_weighted_input * weight
    broadcast_scaled_clamped_weighted_input = tl.broadcast_to(scaled_clamped_weighted_input, [XBLOCK, RBLOCK])
    masked_broadcast = tl.where(xmask, broadcast_scaled_clamped_weighted_input, 0)
    sum_masked_broadcast = tl.sum(masked_broadcast, 1)[:, None]
    broadcast_clamped_weighted_input = tl.broadcast_to(clamped_weighted_input * weight, [XBLOCK, RBLOCK])
    masked_broadcast_clamped = tl.where(xmask, broadcast_clamped_weighted_input, 0)
    sum_masked_broadcast_clamped = tl.sum(masked_broadcast_clamped, 1)[:, None]
    scaled_variance = variance_evict * weight
    clamped_weighted_variance = clamped_weight_grad * scaled_variance
    input_evict_scaled = sum_masked_broadcast_clamped * input_evict
    input_evict_diff = input_evict_scaled - sum_masked_broadcast
    scaled_input_evict_diff = input_evict_diff * variance_evict
    cubed_scaled_input_evict_diff = scaled_input_evict_diff * scaled_input_evict_diff * scaled_input_evict_diff
    scale_factor = 0.015625
    scaled_cubed_input_evict_diff = cubed_scaled_input_evict_diff * scale_factor
    scaled_input_grad = input_grad * scaled_cubed_input_evict_diff
    input_evict_sum = input_evict_scaled + scaled_input_grad
    neg_scaled_cubed_input_evict_diff = -scaled_cubed_input_evict_diff
    neg_scaled_input_evict = neg_scaled_cubed_input_evict_diff * input_evict
    input_evict_product = sum_masked_broadcast_clamped * variance_evict
    scaled_input_evict_product = input_evict_product * scale_factor
    adjusted_input_evict = neg_scaled_input_evict - scaled_input_evict_product
    final_output = input_evict_sum + adjusted_input_evict

    tl.store(output_grad_ptr + (r_block_index + 64 * x_block_index), weighted_sum, xmask)
    tl.store(output_ptr + (r_block_index + 64 * x_block_index), final_output, xmask)