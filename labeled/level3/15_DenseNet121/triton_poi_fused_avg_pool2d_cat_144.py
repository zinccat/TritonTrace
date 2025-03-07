# From: 15_DenseNet121

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_avg_pool2d_cat_144poi_fused_avg_pool2d_cat_144(input_ptr, output_ptr0, output_ptr1, output_ptr2, output_ptr3, output_ptr4, output_ptr5, output_ptr6, output_ptr7, output_ptr8, output_ptr9, num_elements, BLOCK_SIZE : tl.constexpr):
    num_elements = 250880
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    
    col_index = block_indices % 7
    row_index = block_indices // 7
    linear_index = block_indices
    row_col_index = block_indices % 25088
    batch_index = block_indices // 25088
    
    value0 = tl.load(input_ptr + (2 * col_index + 28 * row_index), valid_mask, eviction_policy='evict_last')
    value1 = tl.load(input_ptr + (1 + 2 * col_index + 28 * row_index), valid_mask, eviction_policy='evict_last')
    value3 = tl.load(input_ptr + (14 + 2 * col_index + 28 * row_index), valid_mask, eviction_policy='evict_last')
    value5 = tl.load(input_ptr + (15 + 2 * col_index + 28 * row_index), valid_mask, eviction_policy='evict_last')
    
    sum1 = value1 + value0
    sum2 = value3 + sum1
    sum3 = value5 + sum2
    
    avg_value = sum3 * 0.25
    
    tl.store(output_ptr0 + (linear_index), avg_value, valid_mask)
    tl.store(output_ptr1 + (row_col_index + 37632 * batch_index), avg_value, valid_mask)
    tl.store(output_ptr2 + (row_col_index + 39200 * batch_index), avg_value, valid_mask)
    tl.store(output_ptr3 + (row_col_index + 40768 * batch_index), avg_value, valid_mask)
    tl.store(output_ptr4 + (row_col_index + 42336 * batch_index), avg_value, valid_mask)
    tl.store(output_ptr5 + (row_col_index + 43904 * batch_index), avg_value, valid_mask)
    tl.store(output_ptr6 + (row_col_index + 45472 * batch_index), avg_value, valid_mask)
    tl.store(output_ptr7 + (row_col_index + 47040 * batch_index), avg_value, valid_mask)
    tl.store(output_ptr8 + (row_col_index + 48608 * batch_index), avg_value, valid_mask)
    tl.store(output_ptr9 + (row_col_index + 50176 * batch_index), avg_value, valid_mask)