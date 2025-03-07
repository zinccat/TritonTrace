# From: 15_DenseNet121

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_4poi_fused_max_pool2d_with_indices_4(input_ptr, output_ptr_max, output_ptr_indices, total_elements, BLOCK_SIZE: tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    row = (block_indices // 56) % 56
    col = block_indices % 56
    batch = block_indices // 56
    linear_index = block_indices
    batch_index = block_indices // 3136
    linear_index_within_batch = block_indices % 3136
    
    col_offset = (-1) + 2 * row
    zero_mask = tl.full([1], 0, tl.int64)
    col_valid_mask = (col_offset >= zero_mask) & (col_offset < tl.full([1], 112, tl.int64))
    
    col_index = (-1) + 2 * col
    col_index_valid_mask = (col_index >= zero_mask) & (col_index < tl.full([1], 112, tl.int64))
    valid_mask = col_valid_mask & col_index_valid_mask
    
    value1 = tl.load(input_ptr + ((-113) + 2 * col + 224 * batch), valid_mask, eviction_policy='evict_last', other=float("-inf"))
    value2 = tl.load(input_ptr + ((-112) + 2 * col + 224 * batch), valid_mask & (col_index_valid_mask & (col_index == (-112) + 2 * col)), eviction_policy='evict_last', other=float("-inf"))
    max_val = triton_helpers.maximum(value2, value1)
    
    value3 = tl.load(input_ptr + ((-111) + 2 * col + 224 * batch), valid_mask & (col_index_valid_mask & (col_index == (-111) + 2 * col)), eviction_policy='evict_last', other=float("-inf"))
    max_val = triton_helpers.maximum(value3, max_val)
    
    value4 = tl.load(input_ptr + ((-1) + 2 * col + 224 * batch), valid_mask & (col_index_valid_mask & (col_index == (-1) + 2 * col)), eviction_policy='evict_last', other=float("-inf"))
    max_val = triton_helpers.maximum(value4, max_val)
    
    value5 = tl.load(input_ptr + (2 * col + 224 * batch), valid_mask & (col_index_valid_mask & (col_index == 2 * col)), eviction_policy='evict_last', other=float("-inf"))
    max_val = triton_helpers.maximum(value5, max_val)
    
    value6 = tl.load(input_ptr + (1 + 2 * col + 224 * batch), valid_mask & (col_index_valid_mask & (col_index == 1 + 2 * col)), eviction_policy='evict_last', other=float("-inf"))
    max_val = triton_helpers.maximum(value6, max_val)
    
    value7 = tl.load(input_ptr + (111 + 2 * col + 224 * batch), valid_mask & (col_index_valid_mask & (col_index == 111 + 2 * col)), eviction_policy='evict_last', other=float("-inf"))
    max_val = triton_helpers.maximum(value7, max_val)
    
    value8 = tl.load(input_ptr + (112 + 2 * col + 224 * batch), valid_mask & (col_index_valid_mask & (col_index == 112 + 2 * col)), eviction_policy='evict_last', other=float("-inf"))
    max_val = triton_helpers.maximum(value8, max_val)
    
    value9 = tl.load(input_ptr + (113 + 2 * col + 224 * batch), valid_mask & (col_index_valid_mask & (col_index == 113 + 2 * col)), eviction_policy='evict_last', other=float("-inf"))
    max_val = triton_helpers.maximum(value9, max_val)
    
    index1 = tl.full([1], 1, tl.int8)
    index0 = tl.full([1], 0, tl.int8)
    max_index = tl.where(value2 > value1, index1, index0)
    
    index2 = tl.full([1], 2, tl.int8)
    max_index = tl.where(value3 > max_val, index2, max_index)
    
    index3 = tl.full([1], 3, tl.int8)
    max_index = tl.where(value4 > max_val, index3, max_index)
    
    index4 = tl.full([1], 4, tl.int8)
    max_index = tl.where(value5 > max_val, index4, max_index)
    
    index5 = tl.full([1], 5, tl.int8)
    max_index = tl.where(value6 > max_val, index5, max_index)
    
    index6 = tl.full([1], 6, tl.int8)
    max_index = tl.where(value7 > max_val, index6, max_index)
    
    index7 = tl.full([1], 7, tl.int8)
    max_index = tl.where(value8 > max_val, index7, max_index)
    
    index8 = tl.full([1], 8, tl.int8)
    max_index = tl.where(value9 > max_val, index8, max_index)
    
    tl.store(output_ptr_max + (linear_index), max_val, None)
    tl.store(output_ptr_indices + (linear_index_within_batch + 3200 * batch_index), max_index, None)