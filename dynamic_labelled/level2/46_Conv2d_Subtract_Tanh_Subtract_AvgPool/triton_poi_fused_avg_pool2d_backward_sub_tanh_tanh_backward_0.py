# From: 46_Conv2d_Subtract_Tanh_Subtract_AvgPool

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_avg_pool2d_backward_sub_tanh_tanh_backward_0(
    in_out_ptr, input_ptr, kernel_size_0, kernel_size_1, kernel_size_2, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements

    x_coord = index % kernel_size_0
    y_coord = (index // kernel_size_0) % kernel_size_0
    z_coord = index // kernel_size_1
    linear_index = index

    # Calculate the offset for loading the input tensor
    offset_y = -1 * (((0) * ((0) >= (y_coord // 2)) + (y_coord // 2) * ((y_coord // 2) > (0))))
    offset_y_condition = (
        offset_y <= (-1 + (((-1) + (kernel_size_2 // 2)) * (((-1) + (kernel_size_2 // 2)) <= (1 + (y_coord // 2))) + (1 + (y_coord // 2)) * ((1 + (y_coord // 2)) < ((-1) + (kernel_size_2 // 2))))))
    offset_y_final = offset_y + ((-1) + (((-1) + (kernel_size_2 // 2)) * (((-1) + (kernel_size_2 // 2)) <= (1 + (y_coord // 2))) + (1 + (y_coord // 2)) * ((1 + (y_coord // 2)) < ((-1) + (kernel_size_2 // 2))))) * (offset_y < (((0) * ((0) >= (y_coord // 2)) + (y_coord // 2) * ((y_coord // 2) > (0)))))

    offset_z = z_coord * (kernel_size_2 // 2) * (kernel_size_2 // 2)
    offset_y_scaled = (kernel_size_2 // 2) * offset_y_final
    offset_z_scaled = (-2) * z_coord * (kernel_size_2 // 2)

    offset_x = ((0) * ((0) >= (x_coord // 2)) + (x_coord // 2) * ((x_coord // 2) > (0)))
    offset_x_condition = (
        offset_x <= (-1 + (((-1) + (kernel_size_2 // 2)) * (((-1) + (kernel_size_2 // 2)) <= (1 + (x_coord // 2))) + (1 + (x_coord // 2)) * ((1 + (x_coord // 2)) < ((-1) + (kernel_size_2 // 2))))))
    offset_x_final = offset_x + ((-1) + (((-1) + (kernel_size_2 // 2)) * (((-1) + (kernel_size_2 // 2)) <= (1 + (x_coord // 2))) + (1 + (x_coord // 2)) * ((1 + (x_coord // 2)) < ((-1) + (kernel_size_2 // 2))))) * (offset_x < (((0) * ((0) >= (x_coord // 2)) + (x_coord // 2) * ((x_coord // 2) > (0)))))

    total_offset = offset_z + offset_y_scaled + offset_z_scaled + offset_x_final
    tmp0 = tl.load(input_ptr + (z_coord + total_offset), mask, eviction_policy='evict_last')

    tmp11 = tl.load(in_out_ptr + (linear_index), mask, eviction_policy='evict_last')
    tmp1 = tmp0 / 4

    y_condition = ((0) * ((0) >= (y_coord // 2)) + (y_coord // 2) * ((y_coord // 2) > (0)))
    y_range = (((-1) + (kernel_size_2 // 2)) * (((-1) + (kernel_size_2 // 2)) <= (1 + (y_coord // 2))) + (1 + (y_coord // 2)) * ((1 + (y_coord // 2)) < ((-1) + (kernel_size_2 // 2))))
    y_valid = y_condition < y_range

    x_condition = ((0) * ((0) >= (x_coord // 2)) + (x_coord // 2) * ((x_coord // 2) > (0)))
    x_range = (((-1) + (kernel_size_2 // 2)) * (((-1) + (kernel_size_2 // 2)) <= (1 + (x_coord // 2))) + (1 + (x_coord // 2)) * ((1 + (x_coord // 2)) < ((-1) + (kernel_size_2 // 2))))
    x_valid = x_condition < x_range

    valid_mask = y_valid & x_valid
    tmp10 = tl.where(valid_mask, tmp1, 0.0)

    half = 0.5
    tmp13 = tmp11 - half
    tanh_result = tl.extra.cuda.libdevice.tanh(tmp13)
    tanh_squared = tanh_result * tanh_result
    one_minus_tanh_squared = 1.0 - tanh_squared

    tmp18 = tmp10 * one_minus_tanh_squared
    tl.store(in_out_ptr + (linear_index), tmp18, mask)