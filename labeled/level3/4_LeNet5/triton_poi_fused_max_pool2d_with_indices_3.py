# From: 4_LeNet5

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_3poi_fused_max_pool2d_with_indices_3(input_ptr, output_ptr_max, output_ptr_indices, num_elements, BLOCK_SIZE : tl.constexpr):
    num_elements = 400
    block_offset = tl.program_id(0) * BLOCK_SIZE
    indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = indices < num_elements
    col_index = indices % 5
    row_index = indices // 5
    linear_index = indices

    value_0 = tl.load(input_ptr + (2 * col_index + 20 * row_index), mask, eviction_policy='evict_last')
    value_1 = tl.load(input_ptr + (1 + 2 * col_index + 20 * row_index), mask, eviction_policy='evict_last')
    value_7 = tl.load(input_ptr + (10 + 2 * col_index + 20 * row_index), mask, eviction_policy='evict_last')
    value_12 = tl.load(input_ptr + (11 + 2 * col_index + 20 * row_index), mask, eviction_policy='evict_last')

    is_value_1_greater = value_1 > value_0
    index_1 = tl.full([1], 1, tl.int8)
    index_0 = tl.full([1], 0, tl.int8)
    max_index_01 = tl.where(is_value_1_greater, index_1, index_0)
    max_value_01 = triton_helpers.maximum(value_1, value_0)

    is_value_7_greater = value_7 > max_value_01
    index_2 = tl.full([1], 2, tl.int8)
    max_index_012 = tl.where(is_value_7_greater, index_2, max_index_01)
    max_value_012 = triton_helpers.maximum(value_7, max_value_01)

    is_value_12_greater = value_12 > max_value_012
    index_3 = tl.full([1], 3, tl.int8)
    max_index_final = tl.where(is_value_12_greater, index_3, max_index_012)
    max_value_final = triton_helpers.maximum(value_12, max_value_012)

    tl.store(output_ptr_max + (linear_index), max_index_final, mask)
    tl.store(output_ptr_indices + (linear_index), max_value_final, mask)