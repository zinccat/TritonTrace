# From: 23_Softmax

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused__softmax_4(input_ptr0, input_ptr1, input_ptr2, output_ptr0, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    index2 = index
    block_index = (index // 16384)
    
    value0 = tl.load(input_ptr0 + (index2), None)
    value1 = tl.load(input_ptr1 + (block_index), None, eviction_policy='evict_last')
    value4 = tl.load(input_ptr2 + (block_index), None, eviction_policy='evict_last')
    
    subtracted_value = value0 - value1
    exp_value = tl.math.exp(subtracted_value)
    softmax_value = exp_value / value4
    
    tl.store(output_ptr0 + (index2), softmax_value, None)