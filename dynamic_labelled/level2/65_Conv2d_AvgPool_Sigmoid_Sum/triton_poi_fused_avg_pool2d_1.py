# From: 65_Conv2d_AvgPool_Sigmoid_Sum

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_avg_pool2d_1poi_fused_avg_pool2d_1(input_ptr, output_ptr, kernel_size_0, kernel_size_1, kernel_size_2, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    x_coord = (index % kernel_size_0)
    y_coord = ((index // kernel_size_0) % kernel_size_0)
    z_coord = index // kernel_size_1
    linear_index = index

    # Load elements with specific offsets
    element_0 = tl.load(input_ptr + (((-4) * y_coord) + 2 * x_coord + 4 * z_coord + z_coord * kernel_size_2 * kernel_size_2 + ((-4) * kernel_size_2 * z_coord) + 2 * kernel_size_2 * y_coord), mask, eviction_policy='evict_last')
    element_1 = tl.load(input_ptr + (1 + ((-4) * y_coord) + 2 * x_coord + 4 * z_coord + z_coord * kernel_size_2 * kernel_size_2 + ((-4) * kernel_size_2 * z_coord) + 2 * kernel_size_2 * y_coord), mask, eviction_policy='evict_last')
    element_3 = tl.load(input_ptr + ((-2) + kernel_size_2 + ((-4) * y_coord) + 2 * x_coord + 4 * z_coord + z_coord * kernel_size_2 * kernel_size_2 + ((-4) * kernel_size_2 * z_coord) + 2 * kernel_size_2 * y_coord), mask, eviction_policy='evict_last')
    element_5 = tl.load(input_ptr + ((-1) + kernel_size_2 + ((-4) * y_coord) + 2 * x_coord + 4 * z_coord + z_coord * kernel_size_2 * kernel_size_2 + ((-4) * kernel_size_2 * z_coord) + 2 * kernel_size_2 * y_coord), mask, eviction_policy='evict_last')

    # Compute the sum of elements
    sum_2 = element_1 + element_0
    sum_4 = element_3 + sum_2
    sum_6 = element_5 + sum_4

    # Average the sum
    average = 0.25
    result = sum_6 * average

    # Store the result
    tl.store(output_ptr + (linear_index), result, mask)