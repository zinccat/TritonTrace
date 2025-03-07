# From: 49_ConvTranspose3d_Softmax_Sigmoid

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused__softmax_sigmoid_2(input_ptr0, input_ptr1, input_ptr2, output_ptr0, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    global_index = block_indices
    local_index = block_indices % 131072
    block_index = block_indices // 8388608
    
    input_value0 = tl.load(input_ptr0 + (global_index), None)
    input_value1 = tl.load(input_ptr1 + (local_index + (131072 * block_index)), None, eviction_policy='evict_last')
    input_value2 = tl.load(input_ptr2 + (local_index + (131072 * block_index)), None, eviction_policy='evict_last')
    
    exp_input = input_value0 - input_value1
    exp_result = tl.math.exp(exp_input)
    softmax_result = exp_result / input_value2
    
    sigmoid_result = tl.sigmoid(softmax_result)
    tl.store(output_ptr0 + (global_index), sigmoid_result, None)