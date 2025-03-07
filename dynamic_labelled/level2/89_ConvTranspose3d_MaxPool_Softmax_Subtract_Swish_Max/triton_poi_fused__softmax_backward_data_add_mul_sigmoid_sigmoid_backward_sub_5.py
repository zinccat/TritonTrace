# From: 89_ConvTranspose3d_MaxPool_Softmax_Subtract_Swish_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__softmax_backward_data_add_mul_sigmoid_sigmoid_backward_sub_5poi_fused__softmax_backward_data_add_mul_sigmoid_sigmoid_backward_sub_5(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, kernel_size0, kernel_size1, kernel_size2, kernel_size3, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    indices = offset + tl.arange(0, XBLOCK)[:]
    mask = indices < num_elements
    linear_index = indices
    index_mod_k0 = indices % kernel_size0
    index_div_k1 = indices // kernel_size1
    index_mod_k1 = (indices // kernel_size0) % 16
    grad_output = tl.load(in_out_ptr0 + (linear_index), mask, eviction_policy='evict_last')
    input_data = tl.load(in_ptr0 + (index_mod_k0 + kernel_size2 * index_div_k1 * kernel_size3 * kernel_size3), mask, eviction_policy='evict_last')
    softmax_output = tl.load(in_ptr1 + (linear_index), mask, eviction_policy='evict_last')
    max_value = tl.load(in_ptr2 + (index_mod_k1), mask, eviction_policy='evict_last')
    
    neg_grad_output = -grad_output
    softmax_diff = grad_output - max_value
    sigmoid_output = tl.sigmoid(softmax_diff)
    grad_softmax = softmax_output * sigmoid_output
    grad_softmax_diff = softmax_output * softmax_diff
    one = 1.0
    one_minus_sigmoid = one - sigmoid_output
    grad_sigmoid = sigmoid_output * one_minus_sigmoid
    grad_softmax_diff_sigmoid = grad_softmax_diff * grad_sigmoid
    grad_combined = grad_softmax + grad_softmax_diff_sigmoid
    grad_combined_scaled = grad_combined * grad_output
    fused_result = tl.extra.cuda.libdevice.fma(neg_grad_output, input_data, grad_combined_scaled)
    
    tl.store(in_out_ptr0 + (linear_index), fused_result, mask)