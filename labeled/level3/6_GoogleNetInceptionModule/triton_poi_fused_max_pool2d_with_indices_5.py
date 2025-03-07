# From: 6_GoogleNetInceptionModule

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_5poi_fused_max_pool2d_with_indices_5(input_ptr, output_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    index_within_block = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    col_index = (index_within_block // 107520) % 224
    row_index = (index_within_block // 480) % 224
    linear_index = index_within_block
    
    col_index_minus_one = (-1) + col_index
    zero = tl.full([1], 0, tl.int64)
    max_index = tl.full([1], 224, tl.int64)
    
    col_valid = (col_index_minus_one >= zero) & (col_index_minus_one < max_index)
    row_index_minus_one = (-1) + row_index
    row_valid = (row_index_minus_one >= zero) & (row_index_minus_one < max_index)
    valid_indices = col_valid & row_valid
    
    value_at_col_minus_one = tl.load(input_ptr + (-108000 + linear_index), valid_indices, other=float("-inf"))
    value_at_row_minus_one = tl.load(input_ptr + (-107520 + linear_index), col_valid & row_valid, other=float("-inf"))
    max_value = triton_helpers.maximum(value_at_row_minus_one, value_at_col_minus_one)
    
    next_row_index = 1 + row_index
    next_row_valid = (next_row_index >= zero) & (next_row_index < max_index)
    value_at_next_row = tl.load(input_ptr + (-107040 + linear_index), col_valid & next_row_valid, other=float("-inf"))
    max_value = triton_helpers.maximum(value_at_next_row, max_value)
    
    value_at_current_col = tl.load(input_ptr + (linear_index), col_valid & row_valid, other=float("-inf"))
    max_value = triton_helpers.maximum(value_at_current_col, max_value)
    
    value_at_next_col = tl.load(input_ptr + (480 + linear_index), col_valid & next_row_valid, other=float("-inf"))
    max_value = triton_helpers.maximum(value_at_next_col, max_value)
    
    next_col_index = 1 + col_index
    next_col_valid = (next_col_index >= zero) & (next_col_index < max_index)
    value_at_next_col_row_minus_one = tl.load(input_ptr + (107040 + linear_index), next_col_valid & row_valid, other=float("-inf"))
    max_value = triton_helpers.maximum(value_at_next_col_row_minus_one, max_value)
    
    value_at_next_col_current_row = tl.load(input_ptr + (107520 + linear_index), next_col_valid & row_valid, other=float("-inf"))
    max_value = triton_helpers.maximum(value_at_next_col_current_row, max_value)
    
    value_at_next_col_next_row = tl.load(input_ptr + (108000 + linear_index), next_col_valid & next_row_valid, other=float("-inf"))
    max_value = triton_helpers.maximum(value_at_next_col_next_row, max_value)
    
    tl.store(output_ptr + (linear_index), max_value, None)