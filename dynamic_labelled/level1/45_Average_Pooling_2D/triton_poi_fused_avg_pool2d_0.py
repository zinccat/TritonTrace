# From: 45_Average_Pooling_2D

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_avg_pool2d_0poi_fused_avg_pool2d_0(input_ptr, output_ptr, kernel_size_x, kernel_size_y, kernel_size_z, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    x_coord = (index % kernel_size_x)
    y_coord = ((index // kernel_size_x) % kernel_size_x)
    z_coord = index // kernel_size_y
    linear_index = index

    # Load input values with eviction policy
    input_val_0 = tl.load(input_ptr + (3 * x_coord + z_coord * kernel_size_z * kernel_size_z + 3 * kernel_size_z * y_coord), mask, eviction_policy='evict_last')
    input_val_1 = tl.load(input_ptr + (1 + 3 * x_coord + z_coord * kernel_size_z * kernel_size_z + 3 * kernel_size_z * y_coord), mask, eviction_policy='evict_last')
    input_val_2 = tl.load(input_ptr + (2 + 3 * x_coord + z_coord * kernel_size_z * kernel_size_z + 3 * kernel_size_z * y_coord), mask, eviction_policy='evict_last')
    input_val_3 = tl.load(input_ptr + (kernel_size_z + 3 * x_coord + z_coord * kernel_size_z * kernel_size_z + 3 * kernel_size_z * y_coord), mask, eviction_policy='evict_last')
    input_val_4 = tl.load(input_ptr + (1 + kernel_size_z + 3 * x_coord + z_coord * kernel_size_z * kernel_size_z + 3 * kernel_size_z * y_coord), mask, eviction_policy='evict_last')
    input_val_5 = tl.load(input_ptr + (2 + kernel_size_z + 3 * x_coord + z_coord * kernel_size_z * kernel_size_z + 3 * kernel_size_z * y_coord), mask, eviction_policy='evict_last')
    input_val_6 = tl.load(input_ptr + (2 * kernel_size_z + 3 * x_coord + z_coord * kernel_size_z * kernel_size_z + 3 * kernel_size_z * y_coord), mask, eviction_policy='evict_last')
    input_val_7 = tl.load(input_ptr + (1 + 2 * kernel_size_z + 3 * x_coord + z_coord * kernel_size_z * kernel_size_z + 3 * kernel_size_z * y_coord), mask, eviction_policy='evict_last')
    input_val_8 = tl.load(input_ptr + (2 + 2 * kernel_size_z + 3 * x_coord + z_coord * kernel_size_z * kernel_size_z + 3 * kernel_size_z * y_coord), mask, eviction_policy='evict_last')

    # Sum the loaded values
    sum_val_1 = input_val_1 + input_val_0
    sum_val_2 = input_val_2 + sum_val_1
    sum_val_3 = input_val_3 + sum_val_2
    sum_val_4 = input_val_4 + sum_val_3
    sum_val_5 = input_val_5 + sum_val_4
    sum_val_6 = input_val_6 + sum_val_5
    sum_val_7 = input_val_7 + sum_val_6
    sum_val_8 = input_val_8 + sum_val_7

    # Calculate average
    average = sum_val_8 * 0.1111111111111111

    # Store the result
    tl.store(output_ptr + (linear_index), average, mask)