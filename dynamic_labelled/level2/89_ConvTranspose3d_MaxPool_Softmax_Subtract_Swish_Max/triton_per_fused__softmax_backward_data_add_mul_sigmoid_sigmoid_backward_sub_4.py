# From: 89_ConvTranspose3d_MaxPool_Softmax_Subtract_Swish_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__softmax_backward_data_add_mul_sigmoid_sigmoid_backward_sub_4(
    input_grad_ptr, input_data_ptr, input_sigmoid_ptr, output_ptr, kernel_size_0, kernel_size_1, kernel_size_2, 
    input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr
):
    RBLOCK: tl.constexpr = 16
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < input_num_elements
    r_indices = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = r_indices
    x0 = (x_indices % kernel_size_0)
    x1 = x_indices // kernel_size_0
    x3 = x_indices

    grad_input = tl.load(input_grad_ptr + (x0 + kernel_size_1 * r2 * kernel_size_2 * kernel_size_2 + 16 * kernel_size_1 * x1 * kernel_size_2 * kernel_size_2), x_mask, eviction_policy='evict_last', other=0.0)
    input_data = tl.load(input_data_ptr + (x0 + kernel_size_1 * r2 * kernel_size_2 * kernel_size_2 + 16 * kernel_size_1 * x1 * kernel_size_2 * kernel_size_2), x_mask, eviction_policy='evict_last', other=0.0)
    sigmoid_input = tl.load(input_sigmoid_ptr + (r2), None, eviction_policy='evict_last')
    
    diff = input_data - sigmoid_input
    sigmoid_diff = tl.sigmoid(diff)
    grad_input_scaled = grad_input * sigmoid_diff
    grad_input_diff = grad_input * diff
    one = 1.0
    one_minus_sigmoid_diff = one - sigmoid_diff
    grad_input_scaled_diff = grad_input_diff * one_minus_sigmoid_diff
    combined_grad = grad_input_scaled + grad_input_scaled_diff
    scaled_combined_grad = combined_grad * input_data
    
    broadcasted_grad = tl.broadcast_to(scaled_combined_grad, [XBLOCK, RBLOCK])
    masked_grad = tl.where(x_mask, broadcasted_grad, 0)
    summed_grad = tl.sum(masked_grad, 1)[:, None]
    
    tl.store(output_ptr + (x3), summed_grad, x_mask)