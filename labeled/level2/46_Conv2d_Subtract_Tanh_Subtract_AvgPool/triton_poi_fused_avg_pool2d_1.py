# From: 46_Conv2d_Subtract_Tanh_Subtract_AvgPool

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_avg_pool2d_1(input_ptr, output_ptr, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    col_index = block_indices % 15
    row_index = block_indices // 15
    linear_index = block_indices
    
    input_value_0 = tl.load(input_ptr + ((2 * col_index) + (60 * row_index)), None, eviction_policy='evict_last')
    input_value_1 = tl.load(input_ptr + (1 + (2 * col_index) + (60 * row_index)), None, eviction_policy='evict_last')
    input_value_3 = tl.load(input_ptr + (30 + (2 * col_index) + (60 * row_index)), None, eviction_policy='evict_last')
    input_value_5 = tl.load(input_ptr + (31 + (2 * col_index) + (60 * row_index)), None, eviction_policy='evict_last')
    
    sum_01 = input_value_1 + input_value_0
    sum_034 = input_value_3 + sum_01
    sum_056 = input_value_5 + sum_034
    
    avg_pool_factor = 0.25
    avg_pool_result = sum_056 * avg_pool_factor
    
    tl.store(output_ptr + (linear_index), avg_pool_result, None)