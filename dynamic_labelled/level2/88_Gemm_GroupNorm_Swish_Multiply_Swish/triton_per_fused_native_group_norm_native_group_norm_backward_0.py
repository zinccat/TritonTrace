# From: 88_Gemm_GroupNorm_Swish_Multiply_Swish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_native_group_norm_native_group_norm_backward_0(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, 
    out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr
):
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    
    reduced_index = rindex
    expanded_index = xindex
    element_index = xindex % 16
    
    input_grad = tl.load(in_ptr0 + (reduced_index + 64 * expanded_index), xmask, other=0.0)
    input_data = tl.load(in_ptr1 + (expanded_index), xmask)
    mean = tl.load(in_ptr2 + (expanded_index), xmask)
    variance = tl.load(in_ptr3 + (reduced_index + 64 * element_index), xmask, eviction_policy='evict_last', other=0.0)
    inv_std = tl.load(in_ptr4 + (reduced_index + 64 * element_index), xmask, eviction_policy='evict_last', other=0.0)
    grad_output = tl.load(in_ptr5 + (reduced_index + 64 * expanded_index), xmask, other=0.0)
    swish_grad = tl.load(in_ptr6 + (reduced_index + 64 * element_index), xmask, eviction_policy='evict_last', other=0.0)
    
    mean_adjusted = tl.load(in_ptr2 + (expanded_index), xmask, eviction_policy='evict_last')
    input_data_adjusted = tl.load(in_ptr1 + (expanded_index), xmask, eviction_policy='evict_last')
    
    normalized_input = input_grad - input_data
    normalized_input_scaled = normalized_input * mean
    normalized_input_scaled_inv_std = normalized_input_scaled * inv_std
    swish_input = normalized_input_scaled_inv_std + inv_std
    swish_output = tl.sigmoid(swish_input)
    swish_output_scaled = swish_input * swish_output
    swish_grad_scaled = swish_output_scaled * swish_grad
    swish_output_scaled_sigmoid = tl.sigmoid(swish_grad_scaled)
    grad_output_scaled = grad_output * swish_output_scaled_sigmoid
    grad_output_scaled_swish = grad_output * swish_grad_scaled
    one = 1.0
    one_minus_swish_output = one - swish_output_scaled_sigmoid
    swish_output_scaled_sigmoid_scaled = swish_output_scaled_sigmoid * one_minus_swish_output
    grad_output_scaled_swish_scaled = grad_output_scaled_swish * swish_output_scaled_sigmoid_scaled
    grad_output_combined = grad_output_scaled + grad_output_scaled_swish_scaled
    grad_output_combined_scaled = grad_output_combined * inv_std
    swish_output_scaled_grad = grad_output_combined_scaled * swish_output
    swish_input_scaled_grad = grad_output_combined_scaled * swish_input
    one_minus_swish_output_scaled = one - swish_output
    swish_output_scaled_grad_scaled = swish_input_scaled_grad * one_minus_swish_output_scaled
    grad_output_final = swish_output_scaled_grad + swish_output_scaled_grad_scaled
    grad_input_scaled = grad_output_final * input_grad
    grad_input_scaled_inv_std = grad_input_scaled * inv_std
    grad_input_broadcast = tl.broadcast_to(grad_input_scaled_inv_std, [XBLOCK, RBLOCK])
    grad_input_masked = tl.where(xmask, grad_input_broadcast, 0)
    grad_input_sum = tl.sum(grad_input_masked, 1)[:, None]
    
    grad_input_scaled_broadcast = tl.broadcast_to(grad_input_scaled_inv_std, [XBLOCK, RBLOCK])
    grad_input_scaled_masked = tl.where(xmask, grad_input_scaled_broadcast, 0)
    grad_input_scaled_sum = tl.sum(grad_input_scaled_masked, 1)[:, None]
    
    mean_scaled = mean_adjusted * inv_std
    grad_input_combined = grad_output_final * mean_scaled
    grad_input_adjusted = grad_input_scaled_sum * input_data_adjusted
    grad_input_adjusted_diff = grad_input_adjusted - grad_input_sum
    grad_input_adjusted_scaled = grad_input_adjusted_diff * mean_adjusted
    grad_input_adjusted_scaled_cubed = grad_input_adjusted_scaled * grad_input_adjusted_scaled * grad_input_adjusted_scaled
    scaling_factor = 0.015625
    grad_input_adjusted_scaled_scaled = grad_input_adjusted_scaled_cubed * scaling_factor
    grad_input_adjusted_scaled_input = input_grad * grad_input_adjusted_scaled_scaled
    grad_input_adjusted_combined = grad_input_combined + grad_input_adjusted_scaled_input
    grad_input_adjusted_scaled_neg = -grad_input_adjusted_scaled_scaled
    grad_input_adjusted_scaled_input_neg = grad_input_adjusted_scaled_neg * input_data_adjusted
    grad_input_adjusted_scaled_mean = grad_input_scaled_sum * mean_adjusted
    grad_input_adjusted_scaled_mean_scaled = grad_input_adjusted_scaled_mean * scaling_factor
    grad_input_adjusted_scaled_input_neg_adjusted = grad_input_adjusted_scaled_input_neg - grad_input_adjusted_scaled_mean_scaled
    grad_input_adjusted_final = grad_input_adjusted_combined + grad_input_adjusted_scaled_input_neg_adjusted
    
    tl.store(out_ptr0 + (reduced_index + 64 * expanded_index), swish_input, xmask)
    tl.store(in_out_ptr0 + (reduced_index + 64 * expanded_index), grad_input_adjusted_final, xmask)