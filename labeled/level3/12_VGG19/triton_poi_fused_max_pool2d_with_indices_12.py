# From: 12_VGG19

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_12poi_fused_max_pool2d_with_indices_12(input_ptr, output_ptr_max, output_ptr_indices, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    col_index = block_indices % 128
    row_index = (block_indices // 128) % 56
    channel_index = block_indices // 7168
    linear_index = block_indices
    
    input_offset_base = col_index + 256 * row_index + 28672 * channel_index
    
    input_value_0 = tl.load(input_ptr + input_offset_base, None)
    input_value_1 = tl.load(input_ptr + (128 + input_offset_base), None)
    input_value_3 = tl.load(input_ptr + (14336 + input_offset_base), None)
    input_value_5 = tl.load(input_ptr + (14464 + input_offset_base), None)
    
    max_value_2 = triton_helpers.maximum(input_value_1, input_value_0)
    max_value_4 = triton_helpers.maximum(input_value_3, max_value_2)
    max_value_6 = triton_helpers.maximum(input_value_5, max_value_4)
    
    index_7 = input_value_1 > input_value_0
    index_8 = tl.full([1], 1, tl.int8)
    index_9 = tl.full([1], 0, tl.int8)
    index_10 = tl.where(index_7, index_8, index_9)
    
    index_11 = input_value_3 > max_value_2
    index_12 = tl.full([1], 2, tl.int8)
    index_13 = tl.where(index_11, index_12, index_10)
    
    index_14 = input_value_5 > max_value_4
    index_15 = tl.full([1], 3, tl.int8)
    index_16 = tl.where(index_14, index_15, index_13)
    
    tl.store(output_ptr_max + linear_index, max_value_6, None)
    tl.store(output_ptr_indices + linear_index, index_16, None)