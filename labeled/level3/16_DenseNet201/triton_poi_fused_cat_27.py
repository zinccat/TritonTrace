# From: 16_DenseNet201

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_27poi_fused_cat_27(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, input_ptr5, input_ptr6, 
    output_ptr0, num_elements, BLOCK_SIZE : tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    block_row = (block_indices // 3136) % 256
    block_col = block_indices % 3136
    block_depth = block_indices // 802816
    linear_index = block_indices
    
    row_index = block_row
    tl.full([1], 0, tl.int64)
    
    threshold_64 = tl.full([1], 64, tl.int64)
    condition_64 = row_index < threshold_64
    value_0 = tl.load(input_ptr0 + (block_col + 3136 * row_index + 200704 * block_depth), condition_64, other=0.0)
    
    condition_96 = (row_index >= threshold_64) & (row_index < tl.full([1], 96, tl.int64))
    value_1 = tl.load(input_ptr1 + (block_col + 3136 * ((-64) + row_index) + 100352 * block_depth), condition_96, other=0.0)
    
    condition_128 = (row_index >= tl.full([1], 96, tl.int64)) & (row_index < tl.full([1], 128, tl.int64))
    value_2 = tl.load(input_ptr2 + (block_col + 3136 * ((-96) + row_index) + 100352 * block_depth), condition_128, other=0.0)
    
    condition_160 = (row_index >= tl.full([1], 128, tl.int64)) & (row_index < tl.full([1], 160, tl.int64))
    value_3 = tl.load(input_ptr3 + (block_col + 3136 * ((-128) + row_index) + 100352 * block_depth), condition_160, other=0.0)
    
    condition_192 = (row_index >= tl.full([1], 160, tl.int64)) & (row_index < tl.full([1], 192, tl.int64))
    value_4 = tl.load(input_ptr4 + (block_col + 3136 * ((-160) + row_index) + 100352 * block_depth), condition_192, other=0.0)
    
    condition_224 = (row_index >= tl.full([1], 192, tl.int64)) & (row_index < tl.full([1], 224, tl.int64))
    value_5 = tl.load(input_ptr5 + (block_col + 3136 * ((-192) + row_index) + 100352 * block_depth), condition_224, other=0.0)
    
    condition_256 = row_index >= tl.full([1], 224, tl.int64)
    value_6 = tl.load(input_ptr6 + (block_col + 3136 * ((-224) + row_index) + 100352 * block_depth), condition_256, other=0.0)
    
    result_5 = tl.where(condition_224, value_5, value_6)
    result_4 = tl.where(condition_192, value_4, result_5)
    result_3 = tl.where(condition_160, value_3, result_4)
    result_2 = tl.where(condition_128, value_2, result_3)
    result_1 = tl.where(condition_96, value_1, result_2)
    result_0 = tl.where(condition_64, value_0, result_1)
    
    tl.store(output_ptr0 + (linear_index), result_0, None)