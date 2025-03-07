# From: 89_ConvTranspose3d_MaxPool_Softmax_Subtract_Swish_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__softmax_backward_data_add_mul_sigmoid_sigmoid_backward_sub_5(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, kernel_size_0, kernel_size_1, kernel_size_2, kernel_size_3, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    linear_index = index
    kernel_index_0 = index % kernel_size_0
    kernel_index_2 = index // kernel_size_1
    kernel_index_1 = (index // kernel_size_0) % 16

    grad_output = tl.load(in_out_ptr0 + (linear_index), mask, eviction_policy='evict_last')
    input_data = tl.load(in_ptr0 + (kernel_index_0 + kernel_size_2 * kernel_index_2 * kernel_size_3 * kernel_size_3), mask, eviction_policy='evict_last')
    softmax_output = tl.load(in_ptr1 + (linear_index), mask, eviction_policy='evict_last')
    max_value = tl.load(in_ptr2 + (kernel_index_1), mask, eviction_policy='evict_last')

    neg_grad_output = -grad_output
    adjusted_max = grad_output - max_value
    sigmoid_result = tl.sigmoid(adjusted_max)
    grad_softmax = softmax_output * sigmoid_result
    grad_adjusted_max = softmax_output * adjusted_max
    one = 1.0
    one_minus_sigmoid = one - sigmoid_result
    grad_sigmoid = sigmoid_result * one_minus_sigmoid
    grad_adjusted_max_sigmoid = grad_adjusted_max * grad_sigmoid
    combined_grad = grad_softmax + grad_adjusted_max_sigmoid
    grad_input = combined_grad * grad_output
    fused_result = tl.extra.cuda.libdevice.fma(neg_grad_output, input_data, grad_input)

    tl.store(in_out_ptr0 + (linear_index), fused_result, mask)