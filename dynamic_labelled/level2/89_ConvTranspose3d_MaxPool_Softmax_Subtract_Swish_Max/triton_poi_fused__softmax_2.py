# From: 89_ConvTranspose3d_MaxPool_Softmax_Subtract_Swish_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__softmax_2(output_ptr, input_ptr1, input_ptr2, kernel_size0, kernel_size1, kernel_size2, kernel_size3, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    indices = offset + tl.arange(0, XBLOCK)[:]
    mask = indices < num_elements
    linear_index = indices
    index0 = (indices % kernel_size0)
    index2 = indices // kernel_size1
    loaded_output = tl.load(output_ptr + (linear_index), mask, eviction_policy='evict_last')
    loaded_input1 = tl.load(input_ptr1 + (index0 + kernel_size2 * index2 * kernel_size3 * kernel_size3), mask, eviction_policy='evict_last')
    loaded_input2 = tl.load(input_ptr2 + (index0 + kernel_size2 * index2 * kernel_size3 * kernel_size3), mask, eviction_policy='evict_last')
    subtracted = loaded_output - loaded_input1
    exponentiated = tl.math.exp(subtracted)
    normalized = exponentiated / loaded_input2
    tl.store(output_ptr + (linear_index), normalized, mask)