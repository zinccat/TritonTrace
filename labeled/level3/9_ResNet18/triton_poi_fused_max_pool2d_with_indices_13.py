# From: 9_ResNet18

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_13poi_fused_max_pool2d_with_indices_13(input_ptr, output_ptr_max, output_ptr_indices, num_elements, BLOCK_SIZE: tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    height_index = (block_indices // 3584) % 56
    width_index = (block_indices // 64) % 56
    channel_index = block_indices % 64
    batch_index = block_indices // 3584
    linear_index = block_indices
    
    height_offset = (-1) + 2 * height_index
    zero_mask = tl.full([1], 0, tl.int64)
    height_mask = height_offset >= zero_mask
    max_height = tl.full([1], 112, tl.int64)
    height_bound = height_offset < max_height
    valid_height = height_mask & height_bound
    
    width_offset = (-1) + 2 * width_index
    width_mask = width_offset >= zero_mask
    width_bound = width_offset < max_height
    valid_width = width_mask & width_bound
    valid_position = valid_height & valid_width
    
    value1 = tl.load(input_ptr + ((-7232) + channel_index + 128 * width_index + 14336 * batch_index), valid_position, other=float("-inf"))
    value2 = tl.load(input_ptr + ((-7168) + channel_index + 128 * width_index + 14336 * batch_index), valid_position & (width_offset >= zero_mask) & (width_offset < max_height), other=float("-inf"))
    max_value = triton_helpers.maximum(value2, value1)
    
    value3 = tl.load(input_ptr + ((-7104) + channel_index + 128 * width_index + 14336 * batch_index), valid_position & ((-1 + 2 * width_index) >= zero_mask) & ((-1 + 2 * width_index) < max_height), other=float("-inf"))
    max_value = triton_helpers.maximum(value3, max_value)
    
    value4 = tl.load(input_ptr + ((-64) + channel_index + 128 * width_index + 14336 * batch_index), valid_position & (width_offset >= zero_mask) & (width_offset < max_height) & (height_offset >= zero_mask) & (height_offset < max_height), other=float("-inf"))
    max_value = triton_helpers.maximum(value4, max_value)
    
    value5 = tl.load(input_ptr + (channel_index + 128 * width_index + 14336 * batch_index), valid_position & (width_offset >= zero_mask) & (width_offset < max_height) & (height_offset >= zero_mask) & (height_offset < max_height), other=float("-inf"))
    max_value = triton_helpers.maximum(value5, max_value)
    
    value6 = tl.load(input_ptr + (64 + channel_index + 128 * width_index + 14336 * batch_index), valid_position & (width_offset >= zero_mask) & (width_offset < max_height) & (height_offset >= zero_mask) & (height_offset < max_height), other=float("-inf"))
    max_value = triton_helpers.maximum(value6, max_value)
    
    value7 = tl.load(input_ptr + (7104 + channel_index + 128 * width_index + 14336 * batch_index), valid_position & (width_offset >= zero_mask) & (width_offset < max_height) & (height_offset >= zero_mask) & (height_offset < max_height), other=float("-inf"))
    max_value = triton_helpers.maximum(value7, max_value)
    
    value8 = tl.load(input_ptr + (7168 + channel_index + 128 * width_index + 14336 * batch_index), valid_position & (width_offset >= zero_mask) & (width_offset < max_height) & (height_offset >= zero_mask) & (height_offset < max_height), other=float("-inf"))
    max_value = triton_helpers.maximum(value8, max_value)
    
    value9 = tl.load(input_ptr + (7232 + channel_index + 128 * width_index + 14336 * batch_index), valid_position & (width_offset >= zero_mask) & (width_offset < max_height) & (height_offset >= zero_mask) & (height_offset < max_height), other=float("-inf"))
    max_value = triton_helpers.maximum(value9, max_value)
    
    index1 = value2 > value1
    index2 = tl.full([1], 1, tl.int8)
    index3 = tl.full([1], 0, tl.int8)
    max_index = tl.where(index1, index2, index3)
    
    index4 = value3 > max_value
    index5 = tl.full([1], 2, tl.int8)
    max_index = tl.where(index4, index5, max_index)
    
    index6 = value4 > max_value
    index7 = tl.full([1], 3, tl.int8)
    max_index = tl.where(index6, index7, max_index)
    
    index8 = value5 > max_value
    index9 = tl.full([1], 4, tl.int8)
    max_index = tl.where(index8, index9, max_index)
    
    index10 = value6 > max_value
    index11 = tl.full([1], 5, tl.int8)
    max_index = tl.where(index10, index11, max_index)
    
    index12 = value7 > max_value
    index13 = tl.full([1], 6, tl.int8)
    max_index = tl.where(index12, index13, max_index)
    
    index14 = value8 > max_value
    index15 = tl.full([1], 7, tl.int8)
    max_index = tl.where(index14, index15, max_index)
    
    index16 = value9 > max_value
    index17 = tl.full([1], 8, tl.int8)
    max_index = tl.where(index16, index17, max_index)
    
    tl.store(output_ptr_max + (linear_index), max_value, None)
    tl.store(output_ptr_indices + (linear_index), max_index, None)