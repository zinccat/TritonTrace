# From: 7_GoogleNetInceptionV1

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_38poi_fused_max_pool2d_with_indices_38(input_ptr, output_ptr_max, output_ptr_indices, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    y_coord = (block_indices // 7168) % 14
    x_coord = (block_indices // 512) % 14
    linear_index = block_indices
    
    y_valid = (-1 + y_coord) >= 0
    y_valid &= (-1 + y_coord) < 14
    
    x_valid = (-1 + x_coord) >= 0
    x_valid &= (-1 + x_coord) < 14
    
    valid_indices = y_valid & x_valid
    
    value1 = tl.load(input_ptr + (-7680 + linear_index), valid_indices, other=float("-inf"))
    value2 = tl.load(input_ptr + (-7168 + linear_index), y_valid & x_valid, other=float("-inf"))
    max_value = triton_helpers.maximum(value2, value1)
    
    value3 = tl.load(input_ptr + (-6656 + linear_index), y_valid & (1 + x_coord >= 0) & (1 + x_coord < 14), other=float("-inf"))
    max_value = triton_helpers.maximum(value3, max_value)
    
    value4 = tl.load(input_ptr + (-512 + linear_index), y_valid & x_valid & (x_coord >= 0) & (x_coord < 14), other=float("-inf"))
    max_value = triton_helpers.maximum(value4, max_value)
    
    value5 = tl.load(input_ptr + linear_index, y_valid & x_valid & (1 + x_coord >= 0) & (1 + x_coord < 14), other=float("-inf"))
    max_value = triton_helpers.maximum(value5, max_value)
    
    value6 = tl.load(input_ptr + (512 + linear_index), y_valid & x_valid & (2 + x_coord >= 0) & (2 + x_coord < 14), other=float("-inf"))
    max_value = triton_helpers.maximum(value6, max_value)
    
    value7 = tl.load(input_ptr + (6656 + linear_index), y_valid & (1 + x_coord >= 0) & (1 + x_coord < 14) & (1 + y_coord >= 0) & (1 + y_coord < 14), other=float("-inf"))
    max_value = triton_helpers.maximum(value7, max_value)
    
    value8 = tl.load(input_ptr + (7168 + linear_index), y_valid & (1 + y_coord >= 0) & (1 + y_coord < 14) & (x_coord >= 0) & (x_coord < 14), other=float("-inf"))
    max_value = triton_helpers.maximum(value8, max_value)
    
    value9 = tl.load(input_ptr + (7680 + linear_index), y_valid & (1 + y_coord >= 0) & (1 + y_coord < 14) & (1 + x_coord >= 0) & (1 + x_coord < 14), other=float("-inf"))
    max_value = triton_helpers.maximum(value9, max_value)
    
    index1 = tl.where(value2 > value1, 1, 0)
    index2 = tl.where(value3 > max_value, 2, index1)
    index3 = tl.where(value4 > max_value, 3, index2)
    index4 = tl.where(value5 > max_value, 4, index3)
    index5 = tl.where(value6 > max_value, 5, index4)
    index6 = tl.where(value7 > max_value, 6, index5)
    index7 = tl.where(value8 > max_value, 7, index6)
    index8 = tl.where(value9 > max_value, 8, index7)
    
    tl.store(output_ptr_max + linear_index, max_value, None)
    tl.store(output_ptr_indices + linear_index, index8, None)