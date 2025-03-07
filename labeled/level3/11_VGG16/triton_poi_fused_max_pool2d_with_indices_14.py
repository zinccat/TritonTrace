# From: 11_VGG16

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_14poi_fused_max_pool2d_with_indices_14(input_ptr, output_ptr_max, output_ptr_indices, total_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    col_index = block_indices % 256
    row_index = (block_indices // 256) % 28
    channel_index = block_indices // 7168
    linear_index = block_indices
    
    input_offset_base = col_index + 512 * row_index + 28672 * channel_index
    input_offset_base_1 = input_offset_base + 512
    input_offset_base_2 = input_offset_base + 14336
    input_offset_base_3 = input_offset_base_2 + 128
    
    value0 = tl.load(input_ptr + input_offset_base, None)
    value1 = tl.load(input_ptr + input_offset_base_1, None)
    value3 = tl.load(input_ptr + input_offset_base_2, None)
    value5 = tl.load(input_ptr + input_offset_base_3, None)
    
    max_val_01 = triton_helpers.maximum(value1, value0)
    max_val_23 = triton_helpers.maximum(value3, max_val_01)
    max_val_45 = triton_helpers.maximum(value5, max_val_23)
    
    index_01 = tl.full([1], 1, tl.int8)
    index_00 = tl.full([1], 0, tl.int8)
    index_01_greater = tl.where(value1 > value0, index_01, index_00)
    
    index_23 = tl.full([1], 2, tl.int8)
    index_23_greater = tl.where(value3 > max_val_01, index_23, index_01_greater)
    
    index_45 = tl.full([1], 3, tl.int8)
    index_45_greater = tl.where(value5 > max_val_23, index_45, index_23_greater)
    
    tl.store(output_ptr_max + linear_index, max_val_45, None)
    tl.store(output_ptr_indices + linear_index, index_45_greater, None)