# From: 16_DenseNet201

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_21poi_fused_cat_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    block_row = (xindex // 3136) % 192
    block_col = xindex % 3136
    block_depth = xindex // 602112
    linear_index = xindex
    
    block_row_copy = block_row
    tl.full([1], 0, tl.int64)
    
    max_block_row_1 = tl.full([1], 64, tl.int64)
    is_within_block_1 = block_row_copy < max_block_row_1
    value_1 = tl.load(in_ptr0 + (block_col + 3136 * block_row + 200704 * block_depth), is_within_block_1, other=0.0)
    
    is_outside_block_1 = block_row_copy >= max_block_row_1
    max_block_row_2 = tl.full([1], 96, tl.int64)
    is_within_block_2 = block_row_copy < max_block_row_2
    is_within_block_2_and_outside_block_1 = is_outside_block_1 & is_within_block_2
    value_2 = tl.load(in_ptr1 + (block_col + 3136 * ((-64) + block_row) + 100352 * block_depth), is_within_block_2_and_outside_block_1, other=0.0)
    
    is_outside_block_2 = block_row_copy >= max_block_row_2
    max_block_row_3 = tl.full([1], 128, tl.int64)
    is_within_block_3 = block_row_copy < max_block_row_3
    is_within_block_3_and_outside_block_2 = is_outside_block_2 & is_within_block_3
    value_3 = tl.load(in_ptr2 + (block_col + 3136 * ((-96) + block_row) + 100352 * block_depth), is_within_block_3_and_outside_block_2, other=0.0)
    
    is_outside_block_3 = block_row_copy >= max_block_row_3
    max_block_row_4 = tl.full([1], 160, tl.int64)
    is_within_block_4 = block_row_copy < max_block_row_4
    is_within_block_4_and_outside_block_3 = is_outside_block_3 & is_within_block_4
    value_4 = tl.load(in_ptr3 + (block_col + 3136 * ((-128) + block_row) + 100352 * block_depth), is_within_block_4_and_outside_block_3, other=0.0)
    
    is_outside_block_4 = block_row_copy >= max_block_row_4
    max_block_row_5 = tl.full([1], 192, tl.int64)
    is_within_block_5 = block_row_copy < max_block_row_5
    value_5 = tl.load(in_ptr4 + (block_col + 3136 * ((-160) + block_row) + 100352 * block_depth), is_outside_block_4, other=0.0)
    
    final_value_4 = tl.where(is_within_block_4_and_outside_block_3, value_4, value_5)
    final_value_3 = tl.where(is_within_block_3_and_outside_block_2, value_3, final_value_4)
    final_value_2 = tl.where(is_within_block_2_and_outside_block_1, value_2, final_value_3)
    final_value = tl.where(is_within_block_1, value_1, final_value_2)
    
    tl.store(out_ptr0 + (linear_index), final_value, None)