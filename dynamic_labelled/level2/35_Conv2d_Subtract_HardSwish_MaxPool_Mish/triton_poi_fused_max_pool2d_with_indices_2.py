# From: 35_Conv2d_Subtract_HardSwish_MaxPool_Mish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_2poi_fused_max_pool2d_with_indices_2(
    input_ptr, output_ptr_max, output_ptr_indices, kernel_size_0, kernel_size_1, kernel_size_2, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements

    x_coord = index % kernel_size_0
    y_coord = (index // kernel_size_0) % kernel_size_0
    z_coord = index // kernel_size_1
    linear_index = index

    load_0 = tl.load(
        input_ptr + ((-4) * y_coord + 2 * x_coord + 4 * z_coord + z_coord * kernel_size_2 * kernel_size_2 + (-4) * kernel_size_2 * z_coord + 2 * kernel_size_2 * y_coord),
        mask,
        eviction_policy='evict_last'
    )
    load_1 = tl.load(
        input_ptr + (1 + (-4) * y_coord + 2 * x_coord + 4 * z_coord + z_coord * kernel_size_2 * kernel_size_2 + (-4) * kernel_size_2 * z_coord + 2 * kernel_size_2 * y_coord),
        mask,
        eviction_policy='evict_last'
    )
    load_3 = tl.load(
        input_ptr + ((-2) + kernel_size_2 + (-4) * y_coord + 2 * x_coord + 4 * z_coord + z_coord * kernel_size_2 * kernel_size_2 + (-4) * kernel_size_2 * z_coord + 2 * kernel_size_2 * y_coord),
        mask,
        eviction_policy='evict_last'
    )
    load_5 = tl.load(
        input_ptr + ((-1) + kernel_size_2 + (-4) * y_coord + 2 * x_coord + 4 * z_coord + z_coord * kernel_size_2 * kernel_size_2 + (-4) * kernel_size_2 * z_coord + 2 * kernel_size_2 * y_coord),
        mask,
        eviction_policy='evict_last'
    )

    max_1_0 = triton_helpers.maximum(load_1, load_0)
    max_3_2 = triton_helpers.maximum(load_3, max_1_0)
    max_5_4 = triton_helpers.maximum(load_5, max_3_2)

    index_1_gt_0 = load_1 > load_0
    index_1 = tl.full([1], 1, tl.int8)
    index_0 = tl.full([1], 0, tl.int8)
    index_1_or_0 = tl.where(index_1_gt_0, index_1, index_0)

    index_3_gt_2 = load_3 > max_1_0
    index_3 = tl.full([1], 2, tl.int8)
    index_1_or_0_or_2 = tl.where(index_3_gt_2, index_3, index_1_or_0)

    index_5_gt_4 = load_5 > max_3_2
    index_5 = tl.full([1], 3, tl.int8)
    final_index = tl.where(index_5_gt_4, index_5, index_1_or_0_or_2)

    tl.store(output_ptr_max + linear_index, max_5_4, mask)
    tl.store(output_ptr_indices + linear_index, final_index, mask)