# From: 85_Conv2d_GroupNorm_Scale_MaxPool_Clamp

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clamp_ge_le_logical_and_max_pool2d_with_indices_4poi_fused_clamp_ge_le_logical_and_max_pool2d_with_indices_4(
    input_ptr, output_ptr_indices, output_ptr_values, output_ptr_mask, kernel_size_0, kernel_size_1, kernel_size_2, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    x0 = (index % kernel_size_0)
    x1 = ((index // kernel_size_0) % kernel_size_0)
    x2 = index // kernel_size_1
    x3 = index

    input_value_0 = tl.load(
        input_ptr + (((-4) * x1) + 2 * x0 + 4 * x2 + x2 * kernel_size_2 * kernel_size_2 + ((-4) * kernel_size_2 * x2) + 2 * kernel_size_2 * x1),
        mask,
        eviction_policy='evict_last'
    )
    input_value_1 = tl.load(
        input_ptr + (1 + ((-4) * x1) + 2 * x0 + 4 * x2 + x2 * kernel_size_2 * kernel_size_2 + ((-4) * kernel_size_2 * x2) + 2 * kernel_size_2 * x1),
        mask,
        eviction_policy='evict_last'
    )
    input_value_7 = tl.load(
        input_ptr + ((-2) + kernel_size_2 + ((-4) * x1) + 2 * x0 + 4 * x2 + x2 * kernel_size_2 * kernel_size_2 + ((-4) * kernel_size_2 * x2) + 2 * kernel_size_2 * x1),
        mask,
        eviction_policy='evict_last'
    )
    input_value_12 = tl.load(
        input_ptr + ((-1) + kernel_size_2 + ((-4) * x1) + 2 * x0 + 4 * x2 + x2 * kernel_size_2 * kernel_size_2 + ((-4) * kernel_size_2 * x2) + 2 * kernel_size_2 * x1),
        mask,
        eviction_policy='evict_last'
    )

    is_greater_1 = input_value_1 > input_value_0
    mask_1 = tl.full([1], 1, tl.int8)
    mask_0 = tl.full([1], 0, tl.int8)
    max_index_1 = tl.where(is_greater_1, mask_1, mask_0)
    max_value_1 = triton_helpers.maximum(input_value_1, input_value_0)

    is_greater_7 = input_value_7 > max_value_1
    mask_2 = tl.full([1], 2, tl.int8)
    max_index_2 = tl.where(is_greater_7, mask_2, max_index_1)
    max_value_2 = triton_helpers.maximum(input_value_7, max_value_1)

    is_greater_12 = input_value_12 > max_value_2
    mask_3 = tl.full([1], 3, tl.int8)
    max_index_3 = tl.where(is_greater_12, mask_3, max_index_2)
    max_value_3 = triton_helpers.maximum(input_value_12, max_value_2)

    clamp_min = 0.0
    clamp_max = 1.0
    clamped_value = triton_helpers.maximum(max_value_3, clamp_min)
    clamped_value = triton_helpers.minimum(clamped_value, clamp_max)

    is_ge_min = max_value_3 >= clamp_min
    is_le_max = max_value_3 <= clamp_max
    is_within_range = is_ge_min & is_le_max

    tl.store(output_ptr_indices + (x3), max_index_3, mask)
    tl.store(output_ptr_values + (x3), clamped_value, mask)
    tl.store(output_ptr_mask + (x3), is_within_range, mask)