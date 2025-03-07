# From: 16_DenseNet201

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_24poi_fused_cat_24(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, input_ptr5, 
    output_ptr0, num_elements, BLOCK_SIZE : tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    row_index = (block_indices // 3136) % 224
    col_index = block_indices % 3136
    depth_index = block_indices // 702464
    linear_index = block_indices
    
    row_check = row_index
    tl.full([1], 0, tl.int64)
    
    threshold_64 = tl.full([1], 64, tl.int64)
    is_less_than_64 = row_check < threshold_64
    value_0 = tl.load(input_ptr0 + (col_index + 3136 * row_index + 200704 * depth_index), is_less_than_64, other=0.0)
    
    is_greater_equal_64 = row_check >= threshold_64
    threshold_96 = tl.full([1], 96, tl.int64)
    is_between_64_and_96 = is_greater_equal_64 & (row_check < threshold_96)
    value_1 = tl.load(input_ptr1 + (col_index + 3136 * ((-64) + row_index) + 100352 * depth_index), is_between_64_and_96, other=0.0)
    
    is_greater_equal_96 = row_check >= threshold_96
    threshold_128 = tl.full([1], 128, tl.int64)
    is_between_96_and_128 = is_greater_equal_96 & (row_check < threshold_128)
    value_2 = tl.load(input_ptr2 + (col_index + 3136 * ((-96) + row_index) + 100352 * depth_index), is_between_96_and_128, other=0.0)
    
    is_greater_equal_128 = row_check >= threshold_128
    threshold_160 = tl.full([1], 160, tl.int64)
    is_between_128_and_160 = is_greater_equal_128 & (row_check < threshold_160)
    value_3 = tl.load(input_ptr3 + (col_index + 3136 * ((-128) + row_index) + 100352 * depth_index), is_between_128_and_160, other=0.0)
    
    is_greater_equal_160 = row_check >= threshold_160
    threshold_192 = tl.full([1], 192, tl.int64)
    is_between_160_and_192 = is_greater_equal_160 & (row_check < threshold_192)
    value_4 = tl.load(input_ptr4 + (col_index + 3136 * ((-160) + row_index) + 100352 * depth_index), is_between_160_and_192, other=0.0)
    
    is_greater_equal_192 = row_check >= threshold_192
    threshold_224 = tl.full([1], 224, tl.int64)
    is_greater_equal_224 = row_check >= threshold_224
    value_5 = tl.load(input_ptr5 + (col_index + 3136 * ((-192) + row_index) + 100352 * depth_index), is_greater_equal_192, other=0.0)
    
    final_value = tl.where(is_between_160_and_192, value_4, value_5)
    final_value = tl.where(is_between_128_and_160, value_3, final_value)
    final_value = tl.where(is_between_96_and_128, value_2, final_value)
    final_value = tl.where(is_between_64_and_96, value_1, final_value)
    final_value = tl.where(is_less_than_64, value_0, final_value)
    
    tl.store(output_ptr0 + (linear_index), final_value, None)