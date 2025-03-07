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
    
    r_block_index = rindex
    x_block_index = xindex
    x_mod_index = xindex % 16
    
    input_grad = tl.load(in_ptr0 + (r_block_index + 64 * x_block_index), xmask, other=0.0)
    input_data = tl.load(in_ptr1 + (x_block_index), xmask)
    mean = tl.load(in_ptr2 + (x_block_index), xmask)
    variance = tl.load(in_ptr3 + (r_block_index + 64 * x_mod_index), xmask, eviction_policy='evict_last', other=0.0)
    inv_std = tl.load(in_ptr4 + (r_block_index + 64 * x_mod_index), xmask, eviction_policy='evict_last', other=0.0)
    grad_output = tl.load(in_ptr5 + (r_block_index + 64 * x_block_index), xmask, other=0.0)
    weight = tl.load(in_ptr6 + (r_block_index + 64 * x_mod_index), xmask, eviction_policy='evict_last', other=0.0)
    
    mean_evicted = tl.load(in_ptr2 + (x_block_index), xmask, eviction_policy='evict_last')
    input_data_evicted = tl.load(in_ptr1 + (x_block_index), xmask, eviction_policy='evict_last')
    
    normalized_input = input_grad - input_data
    normalized_input_scaled = normalized_input * mean
    normalized_input_scaled_weighted = normalized_input_scaled * variance
    normalized_input_scaled_weighted_inv_std = normalized_input_scaled_weighted + inv_std
    sigmoid_output = tl.sigmoid(normalized_input_scaled_weighted_inv_std)
    sigmoid_output_scaled = normalized_input_scaled_weighted_inv_std * sigmoid_output
    weighted_sigmoid_output = sigmoid_output_scaled * weight
    sigmoid_weighted = tl.sigmoid(weighted_sigmoid_output)
    grad_output_scaled = grad_output * sigmoid_weighted
    one = 1.0
    one_minus_sigmoid_weighted = one - sigmoid_weighted
    grad_output_scaled_one_minus_sigmoid = grad_output * one_minus_sigmoid_weighted
    grad_output_scaled_one_minus_sigmoid_weighted = grad_output_scaled_one_minus_sigmoid * sigmoid_weighted
    combined_grad_output = grad_output_scaled + grad_output_scaled_one_minus_sigmoid_weighted
    combined_grad_output_weighted = combined_grad_output * weight
    combined_grad_output_weighted_sigmoid = combined_grad_output_weighted * sigmoid_output
    combined_grad_output_weighted_input = combined_grad_output_weighted * normalized_input_scaled_weighted_inv_std
    one_minus_sigmoid_output = one - sigmoid_output
    sigmoid_output_scaled_one_minus_sigmoid = combined_grad_output_weighted_input * (sigmoid_output * one_minus_sigmoid_output)
    final_combined_grad_output = combined_grad_output_weighted_sigmoid + sigmoid_output_scaled_one_minus_sigmoid
    final_combined_grad_output_scaled = final_combined_grad_output * input_grad
    final_combined_grad_output_scaled_weighted = final_combined_grad_output_scaled * variance
    
    broadcasted_final_combined_grad_output_scaled_weighted = tl.broadcast_to(final_combined_grad_output_scaled_weighted, [XBLOCK, RBLOCK])
    masked_broadcasted_final_combined_grad_output_scaled_weighted = tl.where(xmask, broadcasted_final_combined_grad_output_scaled_weighted, 0)
    sum_masked_broadcasted_final_combined_grad_output_scaled_weighted = tl.sum(masked_broadcasted_final_combined_grad_output_scaled_weighted, 1)[:, None]
    
    final_combined_grad_output_scaled = final_combined_grad_output * variance
    broadcasted_final_combined_grad_output_scaled = tl.broadcast_to(final_combined_grad_output_scaled, [XBLOCK, RBLOCK])
    masked_broadcasted_final_combined_grad_output_scaled = tl.where(xmask, broadcasted_final_combined_grad_output_scaled, 0)
    sum_masked_broadcasted_final_combined_grad_output_scaled = tl.sum(masked_broadcasted_final_combined_grad_output_scaled, 1)[:, None]
    
    mean_scaled = mean * variance
    final_combined_grad_output_scaled_mean = final_combined_grad_output * mean_scaled
    sum_masked_input_data_evicted = sum_masked_broadcasted_final_combined_grad_output_scaled * input_data_evicted
    sum_masked_input_data_evicted_minus_sum_masked_broadcasted_final_combined_grad_output_scaled = sum_masked_input_data_evicted - sum_masked_broadcasted_final_combined_grad_output_scaled
    sum_masked_input_data_evicted_minus_sum_masked_broadcasted_final_combined_grad_output_scaled_scaled = sum_masked_input_data_evicted_minus_sum_masked_broadcasted_final_combined_grad_output_scaled * mean_scaled
    sum_masked_input_data_evicted_scaled = sum_masked_input_data_evicted * mean_scaled
    sum_masked_input_data_evicted_scaled_scaled = sum_masked_input_data_evicted_scaled * 0.015625
    input_grad_scaled = input_grad * sum_masked_input_data_evicted_scaled_scaled
    final_combined_grad_output_scaled_mean_plus_input_grad_scaled = final_combined_grad_output_scaled_mean + input_grad_scaled
    negative_sum_masked_input_data_evicted_scaled_scaled = -sum_masked_input_data_evicted_scaled_scaled
    negative_sum_masked_input_data_evicted_scaled_scaled_input_data_evicted = negative_sum_masked_input_data_evicted_scaled_scaled * input_data_evicted
    sum_masked_input_data_evicted_scaled_scaled = sum_masked_input_data_evicted_scaled * 0.015625
    negative_sum_masked_input_data_evicted_scaled_scaled_input_data_evicted_minus_sum_masked_input_data_evicted_scaled_scaled = negative_sum_masked_input_data_evicted_scaled_scaled_input_data_evicted - sum_masked_input_data_evicted_scaled_scaled
    final_combined_grad_output_scaled_mean_plus_input_grad_scaled_plus_negative_sum_masked_input_data_evicted_scaled_scaled_input_data_evicted_minus_sum_masked_input_data_evicted_scaled_scaled = final_combined_grad_output_scaled_mean_plus_input_grad_scaled + negative_sum_masked_input_data_evicted_scaled_scaled_input_data_evicted_minus_sum_masked_input_data_evicted_scaled_scaled
    
    tl.store(out_ptr0 + (r_block_index + 64 * x_block_index), normalized_input_scaled_weighted_inv_std, xmask)
    tl.store(in_out_ptr0 + (r_block_index + 64 * x_block_index), final_combined_grad_output_scaled_mean_plus_input_grad_scaled_plus_negative_sum_masked_input_data_evicted_scaled_scaled_input_data_evicted_minus_sum_masked_input_data_evicted_scaled_scaled, xmask)