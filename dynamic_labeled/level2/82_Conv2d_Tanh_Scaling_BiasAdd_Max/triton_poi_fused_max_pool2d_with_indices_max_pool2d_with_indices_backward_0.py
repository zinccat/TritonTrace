# From: 82_Conv2d_Tanh_Scaling_BiasAdd_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_max_pool2d_with_indices_backward_0(
    input_ptr0, input_ptr1, output_ptr0, kernel_size0, kernel_size1, kernel_size2, total_elements, BLOCK_SIZE : tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < total_elements

    x_coord = block_indices % kernel_size0
    y_coord = (block_indices // kernel_size0) % kernel_size0
    z_coord = block_indices // kernel_size1
    y_index = block_indices % kernel_size1
    linear_index = block_indices

    offset_y = (0 * (0 >= (y_coord // 2)) + (y_coord // 2) * ((y_coord // 2) > 0))
    offset_y_limit = (-1 + ((-1 + (kernel_size2 // 2)) * ((-1 + (kernel_size2 // 2)) <= (1 + (y_coord // 2))) + (1 + (y_coord // 2)) * ((1 + (y_coord // 2)) < (-1 + (kernel_size2 // 2)))))

    offset_y_valid = offset_y * (offset_y <= offset_y_limit) + (-1 + ((-1 + (kernel_size2 // 2)) * ((-1 + (kernel_size2 // 2)) <= (1 + (y_coord // 2))) + (1 + (y_coord // 2)) * ((1 + (y_coord // 2)) < (-1 + (kernel_size2 // 2))))) * ((-1 + ((-1 + (kernel_size2 // 2)) * ((-1 + (kernel_size2 // 2)) <= (1 + (y_coord // 2))) + (1 + (y_coord // 2)) * ((1 + (y_coord // 2)) < (-1 + (kernel_size2 // 2))))) < offset_y))

    offset_x = (0 * (0 >= (x_coord // 2)) + (x_coord // 2) * ((x_coord // 2) > 0))
    offset_x_limit = (-1 + ((-1 + (kernel_size2 // 2)) * ((-1 + (kernel_size2 // 2)) <= (1 + (x_coord // 2))) + (1 + (x_coord // 2)) * ((1 + (x_coord // 2)) < (-1 + (kernel_size2 // 2)))))

    offset_x_valid = offset_x * (offset_x <= offset_x_limit) + (-1 + ((-1 + (kernel_size2 // 2)) * ((-1 + (kernel_size2 // 2)) <= (1 + (x_coord // 2))) + (1 + (x_coord // 2)) * ((1 + (x_coord // 2)) < (-1 + (kernel_size2 // 2))))) * ((-1 + ((-1 + (kernel_size2 // 2)) * ((-1 + (kernel_size2 // 2)) <= (1 + (x_coord // 2))) + (1 + (x_coord // 2)) * ((1 + (x_coord // 2)) < (-1 + (kernel_size2 // 2))))) < offset_x))

    input_index0 = (
        z_coord + (-1 * offset_y) + z_coord * (kernel_size2 // 2) * (kernel_size2 // 2) +
        (kernel_size2 // 2) * offset_y_valid + (-2 * z_coord * (kernel_size2 // 2)) +
        offset_x_valid
    )

    input_index1 = (
        z_coord + (-1 * offset_y) + z_coord * (kernel_size2 // 2) * (kernel_size2 // 2) +
        (kernel_size2 // 2) * offset_y_valid + (-2 * z_coord * (kernel_size2 // 2)) +
        offset_x_valid
    )

    input_value0 = tl.load(input_ptr0 + input_index0, valid_mask, eviction_policy='evict_last')
    input_value1 = tl.load(input_ptr1 + input_index1, valid_mask, eviction_policy='evict_last')

    divisor = tl.full([1], 2, tl.int32)
    quotient = tl.where((input_value0 < 0) != (divisor < 0), tl.where(input_value0 % divisor != 0, input_value0 // divisor - 1, input_value0 // divisor), input_value0 // divisor)
    product = quotient * divisor
    remainder = input_value0 - product

    y_offset = 2 * offset_y_valid
    y_offset_adjusted = y_offset + quotient

    x_offset = 2 * offset_x_valid
    x_offset_adjusted = x_offset + remainder

    linear_offset = y_offset_adjusted * kernel_size0 + x_offset_adjusted

    target_index = y_index
    match = linear_offset == target_index

    output_value = tl.where(match, input_value1, 0.0)

    tl.store(output_ptr0 + linear_index, output_value, valid_mask)