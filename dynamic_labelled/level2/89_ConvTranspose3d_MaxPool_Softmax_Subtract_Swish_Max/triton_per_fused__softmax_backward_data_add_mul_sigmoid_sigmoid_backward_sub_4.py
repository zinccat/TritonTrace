# From: 89_ConvTranspose3d_MaxPool_Softmax_Subtract_Swish_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__softmax_backward_data_add_mul_sigmoid_sigmoid_backward_sub_4(
    input_grad_ptr, input_data_ptr, input_sigmoid_ptr, output_ptr, kernel_size_0, kernel_size_1, kernel_size_2, 
    x_num_elements, r_num_elements, XBLOCK: tl.constexpr
):
    RBLOCK: tl.constexpr = 16
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < x_num_elements
    r_indices = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_block_indices = r_indices
    x_index_mod_k0 = (x_indices % kernel_size_0)
    x_index_div_k0 = x_indices // kernel_size_0
    x_full_indices = x_indices
    input_grad = tl.load(
        input_grad_ptr + (x_index_mod_k0 + kernel_size_1 * r_block_indices * kernel_size_2 * kernel_size_2 + 16 * kernel_size_1 * x_index_div_k0 * kernel_size_2 * kernel_size_2), 
        x_mask, 
        eviction_policy='evict_last', 
        other=0.0
    )
    input_data = tl.load(
        input_data_ptr + (x_index_mod_k0 + kernel_size_1 * r_block_indices * kernel_size_2 * kernel_size_2 + 16 * kernel_size_1 * x_index_div_k0 * kernel_size_2 * kernel_size_2), 
        x_mask, 
        eviction_policy='evict_last', 
        other=0.0
    )
    input_sigmoid = tl.load(input_sigmoid_ptr + (r_block_indices), None, eviction_policy='evict_last')
    diff = input_data - input_sigmoid
    sigmoid_diff = tl.sigmoid(diff)
    grad_weighted_sigmoid = input_grad * sigmoid_diff
    grad_weighted_diff = input_grad * diff
    one = 1.0
    one_minus_sigmoid = one - sigmoid_diff
    grad_weighted_one_minus_sigmoid = sigmoid_diff * one_minus_sigmoid
    grad_weighted_diff_one_minus_sigmoid = grad_weighted_diff * grad_weighted_one_minus_sigmoid
    combined_grad = grad_weighted_sigmoid + grad_weighted_diff_one_minus_sigmoid
    weighted_combined_grad = combined_grad * input_data
    broadcasted_grad = tl.broadcast_to(weighted_combined_grad, [XBLOCK, RBLOCK])
    masked_grad = tl.where(x_mask, broadcasted_grad, 0)
    summed_grad = tl.sum(masked_grad, 1)[:, None]
    tl.store(output_ptr + (x_full_indices), summed_grad, x_mask)