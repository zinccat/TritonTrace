# From: 62_Matmul_GroupNorm_LeakyReLU_Sum

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_leaky_relu_leaky_relu_backward_native_group_norm_native_group_norm_backward_0(
    input_grad_ptr, input_ptr1, input_ptr2, input_ptr3, input_ptr4, input_ptr5, 
    output_grad_ptr, output_ptr3, xnumel, rnumel, XBLOCK: tl.constexpr
):
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_block_index = rindex
    x_block_index = xindex
    x_mod_index = xindex % 8

    grad_input = tl.load(input_grad_ptr + (r_block_index + 32 * x_block_index), xmask, other=0.0)
    input1 = tl.load(input_ptr1 + (x_block_index), xmask)
    input2 = tl.load(input_ptr2 + (x_block_index), xmask)
    grad_input2 = tl.load(input_ptr3 + (r_block_index + 32 * x_mod_index), xmask, eviction_policy='evict_last', other=0.0)
    input3 = tl.load(input_ptr4 + (r_block_index + 32 * x_mod_index), xmask, eviction_policy='evict_last', other=0.0)
    grad_input3 = tl.load(input_ptr5 + (r_block_index + 32 * x_block_index), xmask, other=0.0)
    input2_evict = tl.load(input_ptr2 + (x_block_index), xmask, eviction_policy='evict_last')
    input1_evict = tl.load(input_ptr1 + (x_block_index), xmask, eviction_policy='evict_last')

    diff = grad_input - input1
    scaled_diff = diff * input2
    scaled_diff_grad = scaled_diff * grad_input2
    combined_grad = scaled_diff_grad + input3

    zero = 0.0
    positive_mask = combined_grad > zero
    doubled_grad = grad_input3 + grad_input3
    leaky_relu_slope = 0.01
    leaky_grad = doubled_grad * leaky_relu_slope
    leaky_combined_grad = tl.where(positive_mask, doubled_grad, leaky_grad)
    scaled_leaky_grad = leaky_combined_grad * grad_input
    scaled_leaky_grad_input2 = scaled_leaky_grad * grad_input2
    broadcasted_grad = tl.broadcast_to(scaled_leaky_grad_input2, [XBLOCK, RBLOCK])
    masked_broadcasted_grad = tl.where(xmask, broadcasted_grad, 0)
    sum_masked_broadcasted_grad = tl.sum(masked_broadcasted_grad, 1)[:, None]

    scaled_leaky_grad_input2_broadcast = tl.broadcast_to(scaled_leaky_grad_input2, [XBLOCK, RBLOCK])
    masked_scaled_leaky_grad_input2 = tl.where(xmask, scaled_leaky_grad_input2_broadcast, 0)
    sum_masked_scaled_leaky_grad_input2 = tl.sum(masked_scaled_leaky_grad_input2, 1)[:, None]

    input2_scaled_grad_input2 = input2_evict * grad_input2
    combined_leaky_grad_input2 = leaky_combined_grad * input2_scaled_grad_input2
    diff_sum_input1 = sum_masked_scaled_leaky_grad_input2 * input1_evict
    diff_sum_input1_sub = diff_sum_input1 - sum_masked_broadcasted_grad
    diff_sum_input2 = diff_sum_input1_sub * input2_evict
    cubed_input2 = diff_sum_input2 * diff_sum_input2 * diff_sum_input2
    scaling_factor = 0.03125
    scaled_cubed_input2 = cubed_input2 * scaling_factor
    scaled_grad_input = grad_input * scaled_cubed_input2
    combined_scaled_grad_input = combined_leaky_grad_input2 + scaled_grad_input

    neg_scaled_cubed_input2 = -scaled_cubed_input2
    neg_scaled_input1 = neg_scaled_cubed_input2 * input1_evict
    scaled_diff_input2 = sum_masked_scaled_leaky_grad_input2 * input2_evict
    scaled_diff_input2_scaled = scaled_diff_input2 * scaling_factor
    combined_neg_scaled_input1 = neg_scaled_input1 - scaled_diff_input2_scaled
    final_combined_grad = combined_scaled_grad_input + combined_neg_scaled_input1

    tl.store(output_grad_ptr + (r_block_index + 32 * x_block_index), leaky_combined_grad, xmask)
    tl.store(output_ptr3 + (r_block_index + 32 * x_block_index), final_combined_grad, xmask)