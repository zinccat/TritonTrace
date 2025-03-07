# From: 65_Conv2d_AvgPool_Sigmoid_Sum

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_avg_pool2d_backward_sigmoid_sigmoid_backward_1poi_fused_avg_pool2d_backward_sigmoid_sigmoid_backward_1(
    input_ptr, output_ptr, kernel_size_0, kernel_size_1, kernel_size_2, total_elements, BLOCK_SIZE: tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < total_elements

    x_coord = block_indices % kernel_size_0
    y_coord = (block_indices // kernel_size_0) % kernel_size_0
    z_coord = block_indices // kernel_size_1
    linear_index = block_indices

    offset_y = -1 * (((0) * ((0) >= (y_coord // 2)) + (y_coord // 2) * ((y_coord // 2) > (0))))
    offset_y_limit = (-1 + (((-1) + (kernel_size_2 // 2)) * (((-1) + (kernel_size_2 // 2)) <= (1 + (y_coord // 2))) + (1 + (y_coord // 2)) * ((1 + (y_coord // 2)) < ((-1) + (kernel_size_2 // 2)))))

    offset_x = ((0) * ((0) >= (x_coord // 2)) + (x_coord // 2) * ((x_coord // 2) > (0)))
    offset_x_limit = ((-1) + (((-1) + (kernel_size_2 // 2)) * (((-1) + (kernel_size_2 // 2)) <= (1 + (x_coord // 2))) + (1 + (x_coord // 2)) * ((1 + (x_coord // 2)) < ((-1) + (kernel_size_2 // 2)))))

    load_index = (
        z_coord
        + offset_y
        + z_coord * (kernel_size_2 // 2) * (kernel_size_2 // 2)
        + (kernel_size_2 // 2) * offset_y
        + (-2) * z_coord * (kernel_size_2 // 2)
        + offset_x
    )

    loaded_value = tl.load(input_ptr + load_index, valid_mask, eviction_policy='evict_last')
    averaged_value = loaded_value / 4

    y_condition = (offset_y < offset_y_limit)
    x_condition = (offset_x < offset_x_limit)
    valid_condition = y_condition & x_condition

    result_value = tl.where(valid_condition, averaged_value, 0.0)
    tl.store(output_ptr + linear_index, result_value, valid_mask)