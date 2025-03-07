# From: 7_GoogleNetInceptionV1

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_25poi_fused_max_pool2d_with_indices_25(input_ptr, output_ptr_max, output_ptr_indices, total_elements, BLOCK_SIZE : tl.constexpr):
    total_elements = 1505280
    block_offset = tl.program_id(0) * BLOCK_SIZE
    indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = indices < total_elements

    col_index = (indices // 5376) % 28
    row_index = (indices // 192) % 28
    linear_index = indices

    col_index_offset = col_index - 1
    zero_mask = tl.full([1], 0, tl.int64)
    max_col_index = tl.full([1], 28, tl.int64)

    col_valid_mask = (col_index_offset >= zero_mask) & (col_index_offset < max_col_index)
    row_index_offset = row_index - 1
    row_valid_mask = (row_index_offset >= zero_mask) & (row_index_offset < max_col_index)
    valid_position_mask = col_valid_mask & row_valid_mask

    max_val_1 = tl.load(input_ptr + (-5568 + linear_index), valid_position_mask & valid_mask, other=float("-inf"))
    max_val_2 = tl.load(input_ptr + (-5376 + linear_index), (col_valid_mask & (row_index == row_index_offset)) & valid_mask, other=float("-inf"))
    max_val_3 = triton_helpers.maximum(max_val_2, max_val_1)

    next_row_index = row_index + 1
    next_row_valid_mask = (next_row_index >= zero_mask) & (next_row_index < max_col_index)
    max_val_4 = tl.load(input_ptr + (-5184 + linear_index), (col_valid_mask & next_row_valid_mask) & valid_mask, other=float("-inf"))
    max_val_5 = triton_helpers.maximum(max_val_4, max_val_3)

    next_col_index = col_index + 1
    next_col_valid_mask = (next_col_index >= zero_mask) & (next_col_index < max_col_index)
    max_val_6 = tl.load(input_ptr + (-192 + linear_index), (next_col_valid_mask & row_valid_mask) & valid_mask, other=float("-inf"))
    max_val_7 = triton_helpers.maximum(max_val_6, max_val_5)

    max_val_8 = tl.load(input_ptr + linear_index, (next_col_valid_mask & (row_index == row_index_offset)) & valid_mask, other=float("-inf"))
    max_val_9 = triton_helpers.maximum(max_val_8, max_val_7)

    max_val_10 = tl.load(input_ptr + (192 + linear_index), (next_col_valid_mask & next_row_valid_mask) & valid_mask, other=float("-inf"))
    max_val_11 = triton_helpers.maximum(max_val_10, max_val_9)

    max_val_12 = tl.load(input_ptr + (5184 + linear_index), (next_col_valid_mask & row_valid_mask) & valid_mask, other=float("-inf"))
    max_val_13 = triton_helpers.maximum(max_val_12, max_val_11)

    max_val_14 = tl.load(input_ptr + (5376 + linear_index), (next_col_valid_mask & (row_index == row_index_offset)) & valid_mask, other=float("-inf"))
    max_val_15 = triton_helpers.maximum(max_val_14, max_val_13)

    max_val_16 = tl.load(input_ptr + (5568 + linear_index), (next_col_valid_mask & next_row_valid_mask) & valid_mask, other=float("-inf"))
    max_val_17 = triton_helpers.maximum(max_val_16, max_val_15)

    index_1 = (max_val_2 > max_val_1).astype(tl.int8) * 1
    index_2 = (max_val_4 > max_val_3).astype(tl.int8) * 2
    index_3 = (max_val_6 > max_val_5).astype(tl.int8) * 3
    index_4 = (max_val_8 > max_val_7).astype(tl.int8) * 4
    index_5 = (max_val_10 > max_val_9).astype(tl.int8) * 5
    index_6 = (max_val_12 > max_val_11).astype(tl.int8) * 6
    index_7 = (max_val_14 > max_val_13).astype(tl.int8) * 7
    index_8 = (max_val_16 > max_val_15).astype(tl.int8) * 8

    final_index = tl.where(index_1 > index_2, index_1, index_2)
    final_index = tl.where(index_3 > final_index, index_3, final_index)
    final_index = tl.where(index_4 > final_index, index_4, final_index)
    final_index = tl.where(index_5 > final_index, index_5, final_index)
    final_index = tl.where(index_6 > final_index, index_6, final_index)
    final_index = tl.where(index_7 > final_index, index_7, final_index)
    final_index = tl.where(index_8 > final_index, index_8, final_index)

    tl.store(output_ptr_max + linear_index, max_val_17, valid_mask)
    tl.store(output_ptr_indices + linear_index, final_index, valid_mask)