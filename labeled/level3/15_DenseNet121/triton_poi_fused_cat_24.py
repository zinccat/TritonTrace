# From: 15_DenseNet121

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_24poi_fused_cat_24(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, input_ptr5, 
    output_ptr0, xnumel, XBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    row_index = (x_index // 3136) % 224
    col_index = x_index % 3136
    batch_index = x_index // 702464
    linear_index = x_index
    
    row_check = row_index
    tl.full([1], 0, tl.int64)
    
    threshold_64 = tl.full([1], 64, tl.int64)
    condition_64 = row_check < threshold_64
    value_64 = tl.load(input_ptr0 + (col_index + 3136 * row_index + 200704 * batch_index), condition_64, other=0.0)
    
    condition_96 = (row_check >= threshold_64) & (row_check < tl.full([1], 96, tl.int64))
    value_96 = tl.load(input_ptr1 + (col_index + 3136 * (row_index - 64) + 100352 * batch_index), condition_96, other=0.0)
    
    condition_128 = (row_check >= tl.full([1], 96, tl.int64)) & (row_check < tl.full([1], 128, tl.int64))
    value_128 = tl.load(input_ptr2 + (col_index + 3136 * (row_index - 96) + 100352 * batch_index), condition_128, other=0.0)
    
    condition_160 = (row_check >= tl.full([1], 128, tl.int64)) & (row_check < tl.full([1], 160, tl.int64))
    value_160 = tl.load(input_ptr3 + (col_index + 3136 * (row_index - 128) + 100352 * batch_index), condition_160, other=0.0)
    
    condition_192 = (row_check >= tl.full([1], 160, tl.int64)) & (row_check < tl.full([1], 192, tl.int64))
    value_192 = tl.load(input_ptr4 + (col_index + 3136 * (row_index - 160) + 100352 * batch_index), condition_192, other=0.0)
    
    condition_224 = row_check >= tl.full([1], 192, tl.int64)
    value_224 = tl.load(input_ptr5 + (col_index + 3136 * (row_index - 192) + 100352 * batch_index), condition_224, other=0.0)
    
    result_192 = tl.where(condition_192, value_192, value_224)
    result_160 = tl.where(condition_160, value_160, result_192)
    result_128 = tl.where(condition_128, value_128, result_160)
    result_96 = tl.where(condition_96, value_96, result_128)
    result_64 = tl.where(condition_64, value_64, result_96)
    
    tl.store(output_ptr0 + (linear_index), result_64, None)