# From: 85_Conv2d_GroupNorm_Scale_MaxPool_Clamp

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused_clamp_ge_le_logical_and_max_pool2d_with_indices_2(
    input_ptr, output_ptr_indices, output_ptr_values, output_ptr_clamp, num_elements, BLOCK_SIZE: tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    col_index = block_indices % 15
    row_index = block_indices // 15
    batch_index = block_indices // 3600
    linear_index = block_indices % 3600
    global_index = block_indices
    
    value_0 = tl.load(input_ptr + ((2 * col_index) + (60 * row_index)), None, eviction_policy='evict_last')
    value_1 = tl.load(input_ptr + (1 + (2 * col_index) + (60 * row_index)), None, eviction_policy='evict_last')
    value_7 = tl.load(input_ptr + (30 + (2 * col_index) + (60 * row_index)), None, eviction_policy='evict_last')
    value_12 = tl.load(input_ptr + (31 + (2 * col_index) + (60 * row_index)), None, eviction_policy='evict_last')
    
    is_value_1_greater = value_1 > value_0
    max_index_0_1 = tl.full([1], 1, tl.int8)
    min_index_0_1 = tl.full([1], 0, tl.int8)
    max_index_01 = tl.where(is_value_1_greater, max_index_0_1, min_index_0_1)
    max_value_0_1 = triton_helpers.maximum(value_1, value_0)
    
    is_value_7_greater = value_7 > max_value_0_1
    max_index_0_1_7 = tl.full([1], 2, tl.int8)
    max_index_01_7 = tl.where(is_value_7_greater, max_index_0_1_7, max_index_01)
    max_value_0_1_7 = triton_helpers.maximum(value_7, max_value_0_1)
    
    is_value_12_greater = value_12 > max_value_0_1_7
    max_index_0_1_7_12 = tl.full([1], 3, tl.int8)
    max_index_01_7_12 = tl.where(is_value_12_greater, max_index_0_1_7_12, max_index_01_7)
    max_value_0_1_7_12 = triton_helpers.maximum(value_12, max_value_0_1_7)
    
    clamp_min = 0.0
    clamp_max = 1.0
    clamped_value = triton_helpers.maximum(max_value_0_1_7_12, clamp_min)
    clamped_value = triton_helpers.minimum(clamped_value, clamp_max)
    
    is_within_clamp_range = (max_value_0_1_7_12 >= clamp_min) & (max_value_0_1_7_12 <= clamp_max)
    
    tl.store(output_ptr_indices + (linear_index + (3712 * batch_index)), max_index_01_7_12, None)
    tl.store(output_ptr_values + (global_index), clamped_value, None)
    tl.store(output_ptr_clamp + (linear_index + (3712 * batch_index)), is_within_clamp_range, None)