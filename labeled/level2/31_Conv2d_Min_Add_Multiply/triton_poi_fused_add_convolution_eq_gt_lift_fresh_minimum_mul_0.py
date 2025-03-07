# From: 31_Conv2d_Min_Add_Multiply

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused_add_convolution_eq_gt_lift_fresh_minimum_mul_0(
    input_ptr0, input_ptr1, input_ptr2, output_ptr0, output_ptr1, output_ptr2, 
    num_elements, BLOCK_SIZE : tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    index = block_indices
    channel_index = (block_indices // 900) % 16
    batch_index = (block_indices // 14400)
    flattened_index = block_indices % 14400
    
    input_value0 = tl.load(input_ptr0 + (index), None)
    input_value1 = tl.load(input_ptr1 + (channel_index), None, eviction_policy='evict_last')
    input_value2 = tl.load(input_ptr2 + (channel_index), None, eviction_policy='evict_last')
    
    sum_values = input_value0 + input_value1
    min_value = 0.5
    min_result = triton_helpers.minimum(sum_values, min_value)
    
    add_result = min_result + input_value2
    scale_factor = 2.0
    scaled_result = add_result * scale_factor
    
    equality_check = sum_values == min_value
    greater_than_check = sum_values > min_value
    
    tl.store(output_ptr0 + (index), scaled_result, None)
    tl.store(output_ptr1 + (flattened_index + (14464 * batch_index)), equality_check, None)
    tl.store(output_ptr2 + (flattened_index + (14464 * batch_index)), greater_than_check, None)