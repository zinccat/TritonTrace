# From: 65_Conv2d_AvgPool_Sigmoid_Sum

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_avg_pool2d_backward_sigmoid_sigmoid_backward_1(
    input_ptr, output_ptr, kernel_size_0, kernel_size_1, kernel_size_2, num_elements, BLOCK_SIZE: tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements

    x_coord = block_indices % kernel_size_0
    y_coord = (block_indices // kernel_size_0) % kernel_size_0
    z_coord = block_indices // kernel_size_1
    linear_index = block_indices

    # Calculate the offset for loading data
    offset_y = -1 * (((0) * ((0) >= (y_coord // 2)) + (y_coord // 2) * ((y_coord // 2) > (0))))
    offset_y_condition = (offset_y <= (-1 + (((-1) + (kernel_size_2 // 2)) * (((-1) + (kernel_size_2 // 2)) <= (1 + (y_coord // 2))) + (1 + (y_coord // 2)) * ((1 + (y_coord // 2)) < ((-1) + (kernel_size_2 // 2))))))

    offset_x = ((0) * ((0) >= (x_coord // 2)) + (x_coord // 2) * ((x_coord // 2) > (0)))
    offset_x_condition = (offset_x <= (-1 + (((-1) + (kernel_size_2 // 2)) * (((-1) + (kernel_size_2 // 2)) <= (1 + (x_coord // 2))) + (1 + (x_coord // 2)) * ((1 + (x_coord // 2)) < ((-1) + (kernel_size_2 // 2))))))

    load_index = (
        z_coord
        + offset_y
        + z_coord * (kernel_size_2 // 2) * (kernel_size_2 // 2)
        + (kernel_size_2 // 2) * (offset_y * offset_y_condition)
        + (-2) * z_coord * (kernel_size_2 // 2)
        + (offset_x * offset_x_condition)
    )

    # Load data from input pointer
    loaded_data = tl.load(input_ptr + load_index, valid_mask, eviction_policy='evict_last')

    # Average pooling backward operation
    averaged_data = loaded_data / 4

    # Conditions for valid pooling region
    y_condition = ((0) * ((0) >= (y_coord // 2)) + (y_coord // 2) * ((y_coord // 2) > (0)))
    y_range = (((-1) + (kernel_size_2 // 2)) * (((-1) + (kernel_size_2 // 2)) <= (1 + (y_coord // 2))) + (1 + (y_coord // 2)) * ((1 + (y_coord // 2)) < ((-1) + (kernel_size_2 // 2))))
    y_valid = y_condition < y_range

    x_condition = ((0) * ((0) >= (x_coord // 2)) + (x_coord // 2) * ((x_coord // 2) > (0)))
    x_range = (((-1) + (kernel_size_2 // 2)) * (((-1) + (kernel_size_2 // 2)) <= (1 + (x_coord // 2))) + (1 + (x_coord // 2)) * ((1 + (x_coord // 2)) < ((-1) + (kernel_size_2 // 2))))
    x_valid = x_condition < x_range

    valid_pooling_region = y_valid & x_valid

    # Store the result
    result = tl.where(valid_pooling_region, averaged_data, 0.0)
    tl.store(output_ptr + linear_index, result, valid_mask)