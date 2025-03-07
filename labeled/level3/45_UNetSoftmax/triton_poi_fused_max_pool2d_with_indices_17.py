# From: 45_UNetSoftmax

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_17poi_fused_max_pool2d_with_indices_17(input_ptr, output_ptr_max, output_ptr_indices, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    col_index = block_indices % 32
    row_index = block_indices // 32
    linear_index = block_indices
    
    input_val_0 = tl.load(input_ptr + (2 * col_index + 128 * row_index), None, eviction_policy='evict_last')
    input_val_1 = tl.load(input_ptr + (1 + 2 * col_index + 128 * row_index), None, eviction_policy='evict_last')
    input_val_3 = tl.load(input_ptr + (64 + 2 * col_index + 128 * row_index), None, eviction_policy='evict_last')
    input_val_5 = tl.load(input_ptr + (65 + 2 * col_index + 128 * row_index), None, eviction_policy='evict_last')
    
    max_val_2 = triton_helpers.maximum(input_val_1, input_val_0)
    max_val_4 = triton_helpers.maximum(input_val_3, max_val_2)
    max_val_6 = triton_helpers.maximum(input_val_5, max_val_4)
    
    index_val_7 = input_val_1 > input_val_0
    index_val_8 = tl.full([1], 1, tl.int8)
    index_val_9 = tl.full([1], 0, tl.int8)
    index_val_10 = tl.where(index_val_7, index_val_8, index_val_9)
    
    index_val_11 = input_val_3 > max_val_2
    index_val_12 = tl.full([1], 2, tl.int8)
    index_val_13 = tl.where(index_val_11, index_val_12, index_val_10)
    
    index_val_14 = input_val_5 > max_val_4
    index_val_15 = tl.full([1], 3, tl.int8)
    index_val_16 = tl.where(index_val_14, index_val_15, index_val_13)
    
    tl.store(output_ptr_max + (linear_index), max_val_6, None)
    tl.store(output_ptr_indices + (linear_index), index_val_16, None)