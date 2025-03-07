# From: 15_DenseNet121

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_avg_pool2d_cat_30poi_fused_avg_pool2d_cat_30(
    input_ptr, output_ptr0, output_ptr1, output_ptr2, output_ptr3, output_ptr4, output_ptr5, 
    num_elements, BLOCK_SIZE : tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    col_index = block_indices % 28
    row_index = block_indices // 28
    linear_index = block_indices
    flat_index = block_indices % 100352
    batch_index = block_indices // 100352
    
    value0 = tl.load(input_ptr + (2 * col_index + 112 * row_index), None, eviction_policy='evict_last')
    value1 = tl.load(input_ptr + (1 + 2 * col_index + 112 * row_index), None, eviction_policy='evict_last')
    value3 = tl.load(input_ptr + (56 + 2 * col_index + 112 * row_index), None, eviction_policy='evict_last')
    value5 = tl.load(input_ptr + (57 + 2 * col_index + 112 * row_index), None, eviction_policy='evict_last')
    
    sum1 = value1 + value0
    sum2 = value3 + sum1
    sum3 = value5 + sum2
    
    scale_factor = 0.25
    avg_pool_result = sum3 * scale_factor
    
    tl.store(output_ptr0 + (linear_index), avg_pool_result, None)
    tl.store(output_ptr1 + (flat_index + 301056 * batch_index), avg_pool_result, None)
    tl.store(output_ptr2 + (flat_index + 326144 * batch_index), avg_pool_result, None)
    tl.store(output_ptr3 + (flat_index + 351232 * batch_index), avg_pool_result, None)
    tl.store(output_ptr4 + (flat_index + 376320 * batch_index), avg_pool_result, None)
    tl.store(output_ptr5 + (flat_index + 401408 * batch_index), avg_pool_result, None)