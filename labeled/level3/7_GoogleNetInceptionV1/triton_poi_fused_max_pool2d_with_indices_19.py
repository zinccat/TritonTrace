# From: 7_GoogleNetInceptionV1

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_19poi_fused_max_pool2d_with_indices_19(input_ptr, output_ptr_max, output_ptr_indices, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    index_within_block = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = tl.full([BLOCK_SIZE], True, tl.int1)
    
    y_coord = (index_within_block // 3584) % 56
    x_coord = (index_within_block // 64) % 56
    channel = index_within_block % 64
    batch_index = index_within_block // 3584
    linear_index = index_within_block
    
    y_offset = (-1) + 2 * y_coord
    zero_mask = tl.full([1], 0, tl.int64)
    y_valid = (y_offset >= zero_mask) & (y_offset < tl.full([1], 112, tl.int64))
    
    x_offset = (-1) + 2 * x_coord
    x_valid = (x_offset >= zero_mask) & (x_offset < tl.full([1], 112, tl.int64))
    
    valid_mask = y_valid & x_valid
    
    value1 = tl.load(input_ptr + ((-7232) + channel + 128 * x_coord + 14336 * batch_index), valid_mask, other=float("-inf"))
    value2 = tl.load(input_ptr + ((-7168) + channel + 128 * x_coord + 14336 * batch_index), valid_mask, other=float("-inf"))
    max_value = triton_helpers.maximum(value2, value1)
    
    value3 = tl.load(input_ptr + ((-7104) + channel + 128 * x_coord + 14336 * batch_index), valid_mask, other=float("-inf"))
    max_value = triton_helpers.maximum(value3, max_value)
    
    value4 = tl.load(input_ptr + ((-64) + channel + 128 * x_coord + 14336 * batch_index), valid_mask, other=float("-inf"))
    max_value = triton_helpers.maximum(value4, max_value)
    
    value5 = tl.load(input_ptr + (channel + 128 * x_coord + 14336 * batch_index), valid_mask, other=float("-inf"))
    max_value = triton_helpers.maximum(value5, max_value)
    
    value6 = tl.load(input_ptr + (64 + channel + 128 * x_coord + 14336 * batch_index), valid_mask, other=float("-inf"))
    max_value = triton_helpers.maximum(value6, max_value)
    
    value7 = tl.load(input_ptr + (7104 + channel + 128 * x_coord + 14336 * batch_index), valid_mask, other=float("-inf"))
    max_value = triton_helpers.maximum(value7, max_value)
    
    value8 = tl.load(input_ptr + (7168 + channel + 128 * x_coord + 14336 * batch_index), valid_mask, other=float("-inf"))
    max_value = triton_helpers.maximum(value8, max_value)
    
    value9 = tl.load(input_ptr + (7232 + channel + 128 * x_coord + 14336 * batch_index), valid_mask, other=float("-inf"))
    max_value = triton_helpers.maximum(value9, max_value)
    
    index1 = tl.where(value2 > value1, tl.full([1], 1, tl.int8), tl.full([1], 0, tl.int8))
    index2 = tl.where(value3 > max_value, tl.full([1], 2, tl.int8), index1)
    index3 = tl.where(value4 > max_value, tl.full([1], 3, tl.int8), index2)
    index4 = tl.where(value5 > max_value, tl.full([1], 4, tl.int8), index3)
    index5 = tl.where(value6 > max_value, tl.full([1], 5, tl.int8), index4)
    index6 = tl.where(value7 > max_value, tl.full([1], 6, tl.int8), index5)
    index7 = tl.where(value8 > max_value, tl.full([1], 7, tl.int8), index6)
    index8 = tl.where(value9 > max_value, tl.full([1], 8, tl.int8), index7)
    
    tl.store(output_ptr_max + (linear_index), max_value, None)
    tl.store(output_ptr_indices + (linear_index), index8, None)