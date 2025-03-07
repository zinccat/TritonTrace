# From: 49_ConvTranspose3d_Softmax_Sigmoid

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__softmax__softmax_backward_data_sigmoid_backward_0poi_fused__softmax__softmax_backward_data_sigmoid_backward_0(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, output_ptr0, kernel_size0, kernel_size1, kernel_size2, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    element_index = index
    kernel_index0 = index % kernel_size0
    kernel_index2 = index // kernel_size1
    input_value0 = tl.load(input_ptr0 + (element_index), None, eviction_policy='evict_last')
    input_value1 = tl.load(input_ptr1 + (element_index), None, eviction_policy='evict_last')
    input_value6 = tl.load(input_ptr2 + (element_index), None, eviction_policy='evict_last')
    input_value7 = tl.load(input_ptr3 + (kernel_index0 + 8192 * kernel_size2 * kernel_index2), None, eviction_policy='evict_last')
    input_value10 = tl.load(input_ptr4 + (kernel_index0 + 8192 * kernel_size2 * kernel_index2), None, eviction_policy='evict_last')
    constant_one = 1.0
    subtracted_value = constant_one - input_value1
    multiplied_value1 = input_value1 * subtracted_value
    multiplied_value2 = input_value0 * multiplied_value1
    subtracted_value2 = input_value6 - input_value7
    exponentiated_value = tl.math.exp(subtracted_value2)
    divided_value = exponentiated_value / input_value10
    final_value = multiplied_value2 * divided_value
    tl.store(output_ptr0 + (element_index), final_value, None)