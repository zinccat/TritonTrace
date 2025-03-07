# From: 7_GoogleNetInceptionV1

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_29poi_fused_max_pool2d_with_indices_29(
    input_ptr, output_ptr_max, output_ptr_indices, num_elements, BLOCK_SIZE: tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    col_index = (block_indices // 7168) % 28
    row_index = (block_indices // 256) % 28
    linear_index = block_indices
    
    col_min = (-1) + col_index
    zero = tl.full([1], 0, tl.int64)
    col_max = tl.full([1], 28, tl.int64)
    
    col_valid = (col_min >= zero) & (col_min < col_max)
    row_min = (-1) + row_index
    row_valid = (row_min >= zero) & (row_min < col_max)
    valid_indices = col_valid & row_valid
    
    value1 = tl.load(input_ptr + (-7424 + linear_index), valid_indices, other=float("-inf"))
    value2 = tl.load(input_ptr + (-7168 + linear_index), col_valid & (row_index >= zero) & (row_index < col_max), other=float("-inf"))
    max_value = triton_helpers.maximum(value2, value1)
    
    value3 = tl.load(input_ptr + (-6912 + linear_index), col_valid & ((1 + row_index) >= zero) & ((1 + row_index) < col_max), other=float("-inf"))
    max_value = triton_helpers.maximum(value3, max_value)
    
    value4 = tl.load(input_ptr + (-256 + linear_index), (col_index >= zero) & (col_index < col_max) & row_valid, other=float("-inf"))
    max_value = triton_helpers.maximum(value4, max_value)
    
    value5 = tl.load(input_ptr + linear_index, (col_index >= zero) & (col_index < col_max) & (row_index >= zero) & (row_index < col_max), other=float("-inf"))
    max_value = triton_helpers.maximum(value5, max_value)
    
    value6 = tl.load(input_ptr + (256 + linear_index), (col_index >= zero) & (col_index < col_max) & ((1 + row_index) >= zero) & ((1 + row_index) < col_max), other=float("-inf"))
    max_value = triton_helpers.maximum(value6, max_value)
    
    value7 = tl.load(input_ptr + (6912 + linear_index), (1 + col_index) >= zero) & ((1 + col_index) < col_max) & row_valid, other=float("-inf"))
    max_value = triton_helpers.maximum(value7, max_value)
    
    value8 = tl.load(input_ptr + (7168 + linear_index), (1 + col_index) >= zero) & ((1 + col_index) < col_max) & (row_index >= zero) & (row_index < col_max), other=float("-inf"))
    max_value = triton_helpers.maximum(value8, max_value)
    
    value9 = tl.load(input_ptr + (7424 + linear_index), (1 + col_index) >= zero) & ((1 + col_index) < col_max) & ((1 + row_index) >= zero) & ((1 + row_index) < col_max), other=float("-inf"))
    max_value = triton_helpers.maximum(value9, max_value)
    
    index1 = tl.where(value2 > value1, tl.full([1], 1, tl.int8), tl.full([1], 0, tl.int8))
    index2 = tl.where(value3 > max_value, tl.full([1], 2, tl.int8), index1)
    index3 = tl.where(value4 > max_value, tl.full([1], 3, tl.int8), index2)
    index4 = tl.where(value5 > max_value, tl.full([1], 4, tl.int8), index3)
    index5 = tl.where(value6 > max_value, tl.full([1], 5, tl.int8), index4)
    index6 = tl.where(value7 > max_value, tl.full([1], 6, tl.int8), index5)
    index7 = tl.where(value8 > max_value, tl.full([1], 7, tl.int8), index6)
    index8 = tl.where(value9 > max_value, tl.full([1], 8, tl.int8), index7)
    
    tl.store(output_ptr_max + linear_index, max_value, None)
    tl.store(output_ptr_indices + linear_index, index8, None)