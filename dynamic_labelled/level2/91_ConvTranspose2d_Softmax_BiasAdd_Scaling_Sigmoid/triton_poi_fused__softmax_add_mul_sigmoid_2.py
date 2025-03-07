# From: 91_ConvTranspose2d_Softmax_BiasAdd_Scaling_Sigmoid

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__softmax_add_mul_sigmoid_2poi_fused__softmax_add_mul_sigmoid_2(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, output_ptr0, kernel_size0, kernel_size1, kernel_size2, kernel_size3, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    linear_index = index
    kernel_index0 = index % kernel_size0
    kernel_index1 = index // kernel_size1
    batch_index = (index // kernel_size3) % 64

    input_value0 = tl.load(input_ptr0 + (linear_index), mask, eviction_policy='evict_last')
    input_value1 = tl.load(input_ptr1 + (kernel_index0 + kernel_index1 + 4 * kernel_size2 * kernel_index1 + 4 * kernel_index1 * kernel_size2 * kernel_size2), mask, eviction_policy='evict_last')
    input_value2 = tl.load(input_ptr2 + (kernel_index0 + kernel_index1 + 4 * kernel_size2 * kernel_index1 + 4 * kernel_index1 * kernel_size2 * kernel_size2), mask, eviction_policy='evict_last')
    input_value3 = tl.load(input_ptr3 + (batch_index), mask, eviction_policy='evict_last')

    subtracted_value = input_value0 - input_value1
    exp_value = tl.math.exp(subtracted_value)
    softmax_value = exp_value / input_value2
    added_value = softmax_value + input_value3
    scaled_value = added_value * 2.0
    sigmoid_value = tl.sigmoid(scaled_value)

    tl.store(output_ptr0 + (linear_index), sigmoid_value, mask)