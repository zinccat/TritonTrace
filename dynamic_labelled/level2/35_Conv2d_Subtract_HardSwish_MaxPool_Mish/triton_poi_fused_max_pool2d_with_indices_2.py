# From: 35_Conv2d_Subtract_HardSwish_MaxPool_Mish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_2(input_ptr, output_ptr_values, output_ptr_indices, kernel_size_0, kernel_size_1, kernel_size_2, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements

    col_index = block_indices % kernel_size_0
    row_index = (block_indices // kernel_size_0) % kernel_size_0
    depth_index = block_indices // kernel_size_1
    linear_index = block_indices

    load_offset_0 = (-4 * row_index) + 2 * col_index + 4 * depth_index + depth_index * kernel_size_2 * kernel_size_2 + (-4 * kernel_size_2 * depth_index) + 2 * kernel_size_2 * row_index
    load_offset_1 = 1 + load_offset_0
    load_offset_3 = (-2 + kernel_size_2) + load_offset_0
    load_offset_5 = (-1 + kernel_size_2) + load_offset_0

    value_0 = tl.load(input_ptr + load_offset_0, valid_mask, eviction_policy='evict_last')
    value_1 = tl.load(input_ptr + load_offset_1, valid_mask, eviction_policy='evict_last')
    value_3 = tl.load(input_ptr + load_offset_3, valid_mask, eviction_policy='evict_last')
    value_5 = tl.load(input_ptr + load_offset_5, valid_mask, eviction_policy='evict_last')

    max_01 = triton_helpers.maximum(value_1, value_0)
    max_23 = triton_helpers.maximum(value_3, max_01)
    max_45 = triton_helpers.maximum(value_5, max_23)

    index_0 = tl.full([1], 0, tl.int8)
    index_1 = tl.full([1], 1, tl.int8)
    index_3 = tl.full([1], 2, tl.int8)
    index_5 = tl.full([1], 3, tl.int8)

    index_01 = tl.where(value_1 > value_0, index_1, index_0)
    index_23 = tl.where(value_3 > max_01, index_3, index_01)
    index_45 = tl.where(value_5 > max_23, index_5, index_23)

    tl.store(output_ptr_values + (linear_index), max_45, valid_mask)
    tl.store(output_ptr_indices + (linear_index), index_45, valid_mask)