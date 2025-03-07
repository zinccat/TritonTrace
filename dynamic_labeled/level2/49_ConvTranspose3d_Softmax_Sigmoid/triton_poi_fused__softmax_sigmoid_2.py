# From: 49_ConvTranspose3d_Softmax_Sigmoid

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__softmax_sigmoid_2(input_ptr0, input_ptr1, input_ptr2, output_ptr0, kernel_size0, kernel_size1, kernel_size2, total_elements, XBLOCK : tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    element_index = index
    kernel_index0 = index % kernel_size0
    kernel_index2 = index // kernel_size1
    input_value0 = tl.load(input_ptr0 + (element_index), None, eviction_policy='evict_last')
    input_value1 = tl.load(input_ptr1 + (kernel_index0 + 8192 * kernel_size2 * kernel_index2), None, eviction_policy='evict_last')
    input_value2 = tl.load(input_ptr2 + (kernel_index0 + 8192 * kernel_size2 * kernel_index2), None, eviction_policy='evict_last')
    difference = input_value0 - input_value1
    exp_value = tl.math.exp(difference)
    softmax_value = exp_value / input_value2
    sigmoid_value = tl.sigmoid(softmax_value)
    tl.store(output_ptr0 + (element_index), sigmoid_value, None)