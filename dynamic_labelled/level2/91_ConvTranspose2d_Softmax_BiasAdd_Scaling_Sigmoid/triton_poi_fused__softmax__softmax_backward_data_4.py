# From: 91_ConvTranspose2d_Softmax_BiasAdd_Scaling_Sigmoid

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__softmax__softmax_backward_data_4(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, kernel_size0, kernel_size1, kernel_size2, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    linear_index = index
    kernel_index = index % kernel_size0
    batch_index = index // kernel_size1

    input_value = tl.load(in_out_ptr0 + (linear_index), mask, eviction_policy='evict_last')
    softmax_input = tl.load(in_ptr0 + (kernel_index + batch_index + 4 * kernel_size2 * batch_index + 4 * batch_index * kernel_size2 * kernel_size2), mask, eviction_policy='evict_last')
    softmax_denominator = tl.load(in_ptr1 + (kernel_index + batch_index + 4 * kernel_size2 * batch_index + 4 * batch_index * kernel_size2 * kernel_size2), mask, eviction_policy='evict_last')
    gradient = tl.load(in_ptr2 + (kernel_index + batch_index + 4 * kernel_size2 * batch_index + 4 * batch_index * kernel_size2 * kernel_size2), mask, eviction_policy='evict_last')
    bias = tl.load(in_ptr3 + (linear_index), mask, eviction_policy='evict_last')

    exp_input = input_value - softmax_input
    exp_value = tl.math.exp(exp_input)
    softmax_output = exp_value / softmax_denominator
    neg_softmax_output = -softmax_output

    result = tl.extra.cuda.libdevice.fma(neg_softmax_output, gradient, bias)
    tl.store(in_out_ptr0 + (linear_index), result, mask)