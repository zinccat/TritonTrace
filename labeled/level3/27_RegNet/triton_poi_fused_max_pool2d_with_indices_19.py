# From: 27_RegNet

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_19poi_fused_max_pool2d_with_indices_19(input_ptr, output_ptr_max, output_ptr_indices, total_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    col_index = block_indices % 128
    row_index = (block_indices // 128) % 56
    channel_index = block_indices // 7168
    linear_index = block_indices
    
    input_offset_0 = col_index + 256 * row_index + 28672 * channel_index
    input_offset_1 = 128 + col_index + 256 * row_index + 28672 * channel_index
    input_offset_2 = 14336 + col_index + 256 * row_index + 28672 * channel_index
    input_offset_3 = 14464 + col_index + 256 * row_index + 28672 * channel_index
    
    value_0 = tl.load(input_ptr + input_offset_0, None)
    value_1 = tl.load(input_ptr + input_offset_1, None)
    value_2 = tl.load(input_ptr + input_offset_2, None)
    value_3 = tl.load(input_ptr + input_offset_3, None)
    
    max_01 = triton_helpers.maximum(value_1, value_0)
    max_23 = triton_helpers.maximum(value_2, max_01)
    max_45 = triton_helpers.maximum(value_3, max_23)
    
    index_01 = tl.full([1], 1, tl.int8)
    index_00 = tl.full([1], 0, tl.int8)
    index_01_greater = tl.where(value_1 > value_0, index_01, index_00)
    
    index_23 = tl.full([1], 2, tl.int8)
    index_23_greater = tl.where(value_2 > max_01, index_23, index_01_greater)
    
    index_45 = tl.full([1], 3, tl.int8)
    final_index = tl.where(value_3 > max_23, index_45, index_23_greater)
    
    tl.store(output_ptr_max + linear_index, max_45, None)
    tl.store(output_ptr_indices + linear_index, final_index, None)