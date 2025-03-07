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
    row_index = rindex
    col_index = xindex
    col_mod8 = xindex % 8

    grad_input = tl.load(input_grad_ptr + (row_index + 32 * col_index), xmask, other=0.0)
    input1 = tl.load(input_ptr1 + (col_index), xmask)
    input2 = tl.load(input_ptr2 + (col_index), xmask)
    grad_weight = tl.load(input_ptr3 + (row_index + 32 * col_mod8), xmask, eviction_policy='evict_last', other=0.0)
    weight = tl.load(input_ptr4 + (row_index + 32 * col_mod8), xmask, eviction_policy='evict_last', other=0.0)
    grad_input2 = tl.load(input_ptr5 + (row_index + 32 * col_index), xmask, other=0.0)
    input2_evict = tl.load(input_ptr2 + (col_index), xmask, eviction_policy='evict_last')
    input1_evict = tl.load(input_ptr1 + (col_index), xmask, eviction_policy='evict_last')

    diff = grad_input - input1
    scaled_diff = diff * input2
    scaled_weighted_diff = scaled_diff * weight
    combined_grad = scaled_weighted_diff + weight

    zero = 0.0
    positive_mask = combined_grad > zero
    doubled_grad_input2 = grad_input2 + grad_input2
    leaky_relu_slope = 0.01
    leaky_grad = doubled_grad_input2 * leaky_relu_slope
    leaky_combined_grad = tl.where(positive_mask, doubled_grad_input2, leaky_grad)
    scaled_leaky_grad = leaky_combined_grad * grad_input
    weighted_leaky_grad = scaled_leaky_grad * weight
    broadcasted_weighted_leaky_grad = tl.broadcast_to(weighted_leaky_grad, [XBLOCK, RBLOCK])
    masked_broadcasted_weighted_leaky_grad = tl.where(xmask, broadcasted_weighted_leaky_grad, 0)
    sum_masked_broadcasted_weighted_leaky_grad = tl.sum(masked_broadcasted_weighted_leaky_grad, 1)[:, None]

    scaled_leaky_weight = leaky_combined_grad * weight
    broadcasted_scaled_leaky_weight = tl.broadcast_to(scaled_leaky_weight, [XBLOCK, RBLOCK])
    masked_broadcasted_scaled_leaky_weight = tl.where(xmask, broadcasted_scaled_leaky_weight, 0)
    sum_masked_broadcasted_scaled_leaky_weight = tl.sum(masked_broadcasted_scaled_leaky_weight, 1)[:, None]

    scaled_input2_weight = input2_evict * weight
    scaled_leaky_input2_weight = leaky_combined_grad * scaled_input2_weight
    diff_input1_evict = sum_masked_broadcasted_scaled_leaky_weight * input1_evict
    diff_grad_input = diff_input1_evict - sum_masked_broadcasted_weighted_leaky_grad
    scaled_diff_input2 = diff_grad_input * input2_evict
    cubed_input2 = scaled_diff_input2 * input2_evict * input2_evict * input2_evict
    scaling_factor = 0.03125
    scaled_cubed_input2 = cubed_input2 * scaling_factor
    scaled_grad_input_scaled_cubed_input2 = grad_input * scaled_cubed_input2
    combined_scaled_input2 = scaled_leaky_input2_weight + scaled_grad_input_scaled_cubed_input2

    neg_scaled_cubed_input2 = -scaled_cubed_input2
    scaled_neg_input1_evict = neg_scaled_cubed_input2 * input1_evict
    scaled_diff_input2_scaled_factor = sum_masked_broadcasted_scaled_leaky_weight * input2_evict
    scaled_diff_input2_scaled_cubed_input2 = scaled_diff_input2_scaled_factor * scaling_factor
    combined_scaled_neg_input1_evict = scaled_neg_input1_evict - scaled_diff_input2_scaled_cubed_input2
    final_combined_scaled_input2 = combined_scaled_input2 + combined_scaled_neg_input1_evict

    tl.store(output_grad_ptr + (row_index + 32 * col_index), leaky_combined_grad, xmask)
    tl.store(output_ptr3 + (row_index + 32 * col_index), final_combined_scaled_input2, xmask)