# From: 91_ConvTranspose2d_Softmax_BiasAdd_Scaling_Sigmoid

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__softmax__softmax_backward_data_4poi_fused__softmax__softmax_backward_data_4(
    in_out_ptr, input_ptr, scale_ptr, shift_ptr, bias_ptr, kernel_size_0, kernel_size_1, kernel_size_2, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    linear_index = index
    kernel_index = index % kernel_size_0
    batch_index = index // kernel_size_1

    output_value = tl.load(in_out_ptr + (linear_index), mask, eviction_policy='evict_last')
    input_value = tl.load(input_ptr + (kernel_index + batch_index + 4 * kernel_size_2 * batch_index + 4 * batch_index * kernel_size_2 * kernel_size_2), mask, eviction_policy='evict_last')
    scale_value = tl.load(scale_ptr + (kernel_index + batch_index + 4 * kernel_size_2 * batch_index + 4 * batch_index * kernel_size_2 * kernel_size_2), mask, eviction_policy='evict_last')
    shift_value = tl.load(shift_ptr + (kernel_index + batch_index + 4 * kernel_size_2 * batch_index + 4 * batch_index * kernel_size_2 * kernel_size_2), mask, eviction_policy='evict_last')
    bias_value = tl.load(bias_ptr + (linear_index), mask, eviction_policy='evict_last')

    diff = output_value - input_value
    exp_diff = tl.math.exp(diff)
    softmax_grad = exp_diff / scale_value
    neg_softmax_grad = -softmax_grad

    updated_value = tl.extra.cuda.libdevice.fma(neg_softmax_grad, shift_value, bias_value)
    tl.store(in_out_ptr + (linear_index), updated_value, mask)