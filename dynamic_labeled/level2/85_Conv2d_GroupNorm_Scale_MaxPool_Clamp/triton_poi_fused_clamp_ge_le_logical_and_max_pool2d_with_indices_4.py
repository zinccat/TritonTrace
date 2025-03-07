# From: 85_Conv2d_GroupNorm_Scale_MaxPool_Clamp

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clamp_ge_le_logical_and_max_pool2d_with_indices_4(
    input_ptr, output_ptr_indices, output_ptr_clamped_values, output_ptr_clamp_mask, kernel_size_x, kernel_size_y, kernel_size_z, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements

    x_coord = index % kernel_size_x
    y_coord = (index // kernel_size_x) % kernel_size_x
    z_coord = index // kernel_size_y
    linear_index = index

    load_offset_0 = ((-4) * y_coord) + 2 * x_coord + 4 * z_coord + z_coord * kernel_size_z * kernel_size_z + ((-4) * kernel_size_z * z_coord) + 2 * kernel_size_z * y_coord
    load_offset_1 = 1 + load_offset_0
    load_offset_7 = (-2) + kernel_size_z + load_offset_0
    load_offset_12 = (-1) + kernel_size_z + load_offset_0

    value_0 = tl.load(input_ptr + load_offset_0, mask, eviction_policy='evict_last')
    value_1 = tl.load(input_ptr + load_offset_1, mask, eviction_policy='evict_last')
    value_7 = tl.load(input_ptr + load_offset_7, mask, eviction_policy='evict_last')
    value_12 = tl.load(input_ptr + load_offset_12, mask, eviction_policy='evict_last')

    is_value_1_greater = value_1 > value_0
    true_mask = tl.full([1], 1, tl.int8)
    false_mask = tl.full([1], 0, tl.int8)
    max_index_1 = tl.where(is_value_1_greater, true_mask, false_mask)
    max_value_1 = triton_helpers.maximum(value_1, value_0)

    is_value_7_greater = value_7 > max_value_1
    index_7 = tl.full([1], 2, tl.int8)
    max_index_7 = tl.where(is_value_7_greater, index_7, max_index_1)
    max_value_7 = triton_helpers.maximum(value_7, max_value_1)

    is_value_12_greater = value_12 > max_value_7
    index_12 = tl.full([1], 3, tl.int8)
    max_index_12 = tl.where(is_value_12_greater, index_12, max_index_7)
    max_value_12 = triton_helpers.maximum(value_12, max_value_7)

    clamp_min = 0.0
    clamp_max = 1.0
    clamped_value = triton_helpers.maximum(max_value_12, clamp_min)
    clamped_value = triton_helpers.minimum(clamped_value, clamp_max)

    is_within_clamp_range = (max_value_12 >= clamp_min) & (max_value_12 <= clamp_max)

    tl.store(output_ptr_indices + (linear_index), max_index_12, mask)
    tl.store(output_ptr_clamped_values + (linear_index), clamped_value, mask)
    tl.store(output_ptr_clamp_mask + (linear_index), is_within_clamp_range, mask)