# From: 91_ConvTranspose2d_Softmax_BiasAdd_Scaling_Sigmoid

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused__softmax_add_mul_sigmoid_2(input_ptr0, input_ptr1, input_ptr2, input_ptr3, output_ptr0, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    index_3d = block_indices
    index_0 = block_indices % 1089
    index_2 = (block_indices // 69696)
    index_1 = (block_indices // 1089) % 64
    
    input_value0 = tl.load(input_ptr0 + (index_3d), None)
    input_value1 = tl.load(input_ptr1 + (index_0 + (1120 * index_2)), None, eviction_policy='evict_last')
    input_value4 = tl.load(input_ptr2 + (index_0 + (1120 * index_2)), None, eviction_policy='evict_last')
    input_value6 = tl.load(input_ptr3 + (index_1), None, eviction_policy='evict_last')
    
    subtracted_value = input_value0 - input_value1
    exp_value = tl.math.exp(subtracted_value)
    softmax_value = exp_value / input_value4
    added_value = softmax_value + input_value6
    
    scaling_factor = 2.0
    scaled_value = added_value * scaling_factor
    sigmoid_value = tl.sigmoid(scaled_value)
    
    tl.store(output_ptr0 + (index_3d), sigmoid_value, None)