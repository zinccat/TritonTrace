# From: 23_Softmax

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__softmax_4poi_fused__softmax_4(input_ptr0, input_ptr1, input_ptr2, output_ptr0, kernel_size, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    index2 = index
    index1 = index // kernel_size
    
    value0 = tl.load(input_ptr0 + (index2), mask, eviction_policy='evict_last')
    value1 = tl.load(input_ptr1 + (index1), mask, eviction_policy='evict_last')
    value4 = tl.load(input_ptr2 + (index1), mask, eviction_policy='evict_last')
    
    subtracted_value = value0 - value1
    exp_value = tl.math.exp(subtracted_value)
    softmax_value = exp_value / value4
    
    tl.store(output_ptr0 + (index2), softmax_value, mask)