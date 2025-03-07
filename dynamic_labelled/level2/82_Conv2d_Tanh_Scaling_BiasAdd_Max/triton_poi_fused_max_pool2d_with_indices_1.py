# From: 82_Conv2d_Tanh_Scaling_BiasAdd_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_1poi_fused_max_pool2d_with_indices_1(
    input_ptr, output_ptr_max, output_ptr_indices, kernel_size_0, kernel_size_1, kernel_size_2, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements

    x0 = (index % kernel_size_0)
    x1 = ((index // kernel_size_0) % kernel_size_0)
    x2 = index // kernel_size_1
    x3 = index

    load1 = tl.load(
        input_ptr + (((-4) * x1) + 2 * x0 + 4 * x2 + x2 * kernel_size_2 * kernel_size_2 + ((-4) * kernel_size_2 * x2) + 2 * kernel_size_2 * x1),
        mask,
        eviction_policy='evict_last'
    )
    load2 = tl.load(
        input_ptr + (1 + ((-4) * x1) + 2 * x0 + 4 * x2 + x2 * kernel_size_2 * kernel_size_2 + ((-4) * kernel_size_2 * x2) + 2 * kernel_size_2 * x1),
        mask,
        eviction_policy='evict_last'
    )
    load3 = tl.load(
        input_ptr + ((-2) + kernel_size_2 + ((-4) * x1) + 2 * x0 + 4 * x2 + x2 * kernel_size_2 * kernel_size_2 + ((-4) * kernel_size_2 * x2) + 2 * kernel_size_2 * x1),
        mask,
        eviction_policy='evict_last'
    )
    load4 = tl.load(
        input_ptr + ((-1) + kernel_size_2 + ((-4) * x1) + 2 * x0 + 4 * x2 + x2 * kernel_size_2 * kernel_size_2 + ((-4) * kernel_size_2 * x2) + 2 * kernel_size_2 * x1),
        mask,
        eviction_policy='evict_last'
    )

    max1 = triton_helpers.maximum(load2, load1)
    max2 = triton_helpers.maximum(load3, max1)
    max3 = triton_helpers.maximum(load4, max2)

    index1 = load2 > load1
    index_value1 = tl.full([1], 1, tl.int8)
    index_value0 = tl.full([1], 0, tl.int8)
    index_result1 = tl.where(index1, index_value1, index_value0)

    index2 = load3 > max1
    index_value2 = tl.full([1], 2, tl.int8)
    index_result2 = tl.where(index2, index_value2, index_result1)

    index3 = load4 > max2
    index_value3 = tl.full([1], 3, tl.int8)
    index_result3 = tl.where(index3, index_value3, index_result2)

    tl.store(output_ptr_max + (x3), max3, mask)
    tl.store(output_ptr_indices + (x3), index_result3, mask)