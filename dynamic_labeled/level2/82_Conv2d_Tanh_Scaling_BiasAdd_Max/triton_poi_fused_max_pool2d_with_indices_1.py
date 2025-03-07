# From: 82_Conv2d_Tanh_Scaling_BiasAdd_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_1(input_ptr, output_ptr_max, output_ptr_indices, kernel_size_0, kernel_size_1, kernel_size_2, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements

    col_index = block_indices % kernel_size_0
    row_index = (block_indices // kernel_size_0) % kernel_size_0
    depth_index = block_indices // kernel_size_1
    linear_index = block_indices

    offset_base = depth_index * kernel_size_2 * kernel_size_2 + 2 * kernel_size_2 * row_index - 4 * row_index + 4 * depth_index
    offset_0 = offset_base + 2 * col_index
    offset_1 = offset_0 + 1
    offset_3 = offset_0 + ks2 - 2
    offset_5 = offset_0 + ks2 - 1

    value_0 = tl.load(input_ptr + offset_0, valid_mask, eviction_policy='evict_last')
    value_1 = tl.load(input_ptr + offset_1, valid_mask, eviction_policy='evict_last')
    value_3 = tl.load(input_ptr + offset_3, valid_mask, eviction_policy='evict_last')
    value_5 = tl.load(input_ptr + offset_5, valid_mask, eviction_policy='evict_last')

    max_01 = triton_helpers.maximum(value_1, value_0)
    max_23 = triton_helpers.maximum(value_3, max_01)
    max_45 = triton_helpers.maximum(value_5, max_23)

    index_0 = tl.full([1], 0, tl.int8)
    index_1 = tl.full([1], 1, tl.int8)
    index_3 = tl.full([1], 2, tl.int8)
    index_5 = tl.full([1], 3, tl.int8)

    index_max_01 = tl.where(value_1 > value_0, index_1, index_0)
    index_max_23 = tl.where(value_3 > max_01, index_3, index_max_01)
    index_max_45 = tl.where(value_5 > max_23, index_5, index_max_23)

    tl.store(output_ptr_max + linear_index, max_45, valid_mask)
    tl.store(output_ptr_indices + linear_index, index_max_45, valid_mask)