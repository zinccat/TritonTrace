# From: 10_ResNet101

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_10poi_fused_max_pool2d_with_indices_10(input_ptr, output_ptr_max, output_ptr_indices, num_elements, BLOCK_SIZE: tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    index_within_block = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = tl.full([BLOCK_SIZE], True, tl.int1)
    
    # Calculate indices for pooling
    pooled_height_index = (index_within_block // 3584) % 56
    pooled_width_index = (index_within_block // 64) % 56
    input_width_index = index_within_block % 64
    batch_index = index_within_block // 3584
    linear_index = index_within_block
    
    # Calculate temporary variables for pooling
    height_offset = (-1) + 2 * pooled_height_index
    zero_mask = tl.full([1], 0, tl.int64)
    height_valid_mask = (height_offset >= zero_mask) & (height_offset < tl.full([1], 112, tl.int64))
    
    width_offset = (-1) + 2 * pooled_width_index
    width_valid_mask = (width_offset >= zero_mask) & (width_offset < tl.full([1], 112, tl.int64))
    
    valid_mask = height_valid_mask & width_valid_mask
    
    # Load and compare values for max pooling
    value1 = tl.load(input_ptr + ((-7232) + input_width_index + 128 * pooled_width_index + 14336 * batch_index), valid_mask, other=float("-inf"))
    value2 = tl.load(input_ptr + ((-7168) + input_width_index + 128 * pooled_width_index + 14336 * batch_index), valid_mask, other=float("-inf"))
    max_value = triton_helpers.maximum(value2, value1)
    
    value3 = tl.load(input_ptr + ((-7104) + input_width_index + 128 * pooled_width_index + 14336 * batch_index), valid_mask, other=float("-inf"))
    max_value = triton_helpers.maximum(value3, max_value)
    
    value4 = tl.load(input_ptr + ((-64) + input_width_index + 128 * pooled_width_index + 14336 * batch_index), valid_mask, other=float("-inf"))
    max_value = triton_helpers.maximum(value4, max_value)
    
    value5 = tl.load(input_ptr + (input_width_index + 128 * pooled_width_index + 14336 * batch_index), valid_mask, other=float("-inf"))
    max_value = triton_helpers.maximum(value5, max_value)
    
    value6 = tl.load(input_ptr + (64 + input_width_index + 128 * pooled_width_index + 14336 * batch_index), valid_mask, other=float("-inf"))
    max_value = triton_helpers.maximum(value6, max_value)
    
    value7 = tl.load(input_ptr + (7104 + input_width_index + 128 * pooled_width_index + 14336 * batch_index), valid_mask, other=float("-inf"))
    max_value = triton_helpers.maximum(value7, max_value)
    
    value8 = tl.load(input_ptr + (7168 + input_width_index + 128 * pooled_width_index + 14336 * batch_index), valid_mask, other=float("-inf"))
    max_value = triton_helpers.maximum(value8, max_value)
    
    value9 = tl.load(input_ptr + (7232 + input_width_index + 128 * pooled_width_index + 14336 * batch_index), valid_mask, other=float("-inf"))
    max_value = triton_helpers.maximum(value9, max_value)
    
    # Determine indices of max values
    index1 = tl.where(value1 > max_value, tl.full([1], 1, tl.int8), tl.full([1], 0, tl.int8))
    index2 = tl.where(value2 > max_value, tl.full([1], 2, tl.int8), index1)
    index3 = tl.where(value3 > max_value, tl.full([1], 3, tl.int8), index2)
    index4 = tl.where(value4 > max_value, tl.full([1], 4, tl.int8), index3)
    index5 = tl.where(value5 > max_value, tl.full([1], 5, tl.int8), index4)
    index6 = tl.where(value6 > max_value, tl.full([1], 6, tl.int8), index5)
    index7 = tl.where(value7 > max_value, tl.full([1], 7, tl.int8), index6)
    index8 = tl.where(value8 > max_value, tl.full([1], 8, tl.int8), index7)
    max_index = tl.where(value9 > max_value, tl.full([1], 9, tl.int8), index8)
    
    # Store results
    tl.store(output_ptr_max + (linear_index), max_value, None)
    tl.store(output_ptr_indices + (linear_index), max_index, None)