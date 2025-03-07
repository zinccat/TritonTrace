# From: 45_UNetSoftmax

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_4poi_fused_max_pool2d_with_indices_4(input_ptr, output_ptr_max, output_ptr_indices, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    col_index = block_indices % 256
    row_index = block_indices // 256
    linear_index = block_indices
    
    value0 = tl.load(input_ptr + (2 * col_index + 1024 * row_index), None, eviction_policy='evict_last')
    value1 = tl.load(input_ptr + (1 + 2 * col_index + 1024 * row_index), None, eviction_policy='evict_last')
    value3 = tl.load(input_ptr + (512 + 2 * col_index + 1024 * row_index), None, eviction_policy='evict_last')
    value5 = tl.load(input_ptr + (513 + 2 * col_index + 1024 * row_index), None, eviction_policy='evict_last')
    
    max_val_01 = triton_helpers.maximum(value1, value0)
    max_val_23 = triton_helpers.maximum(value3, max_val_01)
    max_val_45 = triton_helpers.maximum(value5, max_val_23)
    
    index_01 = tl.where(value1 > value0, tl.full([1], 1, tl.int8), tl.full([1], 0, tl.int8))
    index_23 = tl.where(value3 > max_val_01, tl.full([1], 2, tl.int8), index_01)
    index_45 = tl.where(value5 > max_val_23, tl.full([1], 3, tl.int8), index_23)
    
    tl.store(output_ptr_max + (linear_index), max_val_45, None)
    tl.store(output_ptr_indices + (linear_index), index_45, None)