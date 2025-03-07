# From: 85_Conv2d_GroupNorm_Scale_MaxPool_Clamp

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_max_pool2d_with_indices_backward_scalar_tensor_where_1poi_fused_max_pool2d_with_indices_max_pool2d_with_indices_backward_scalar_tensor_where_1(
    input_ptr0, input_ptr1, output_ptr0, kernel_size0, kernel_size1, kernel_size2, total_elements, BLOCK_SIZE : tl.constexpr):

    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < total_elements

    x_coord = block_indices % kernel_size0
    y_coord = (block_indices // kernel_size0) % kernel_size0
    z_coord = block_indices // kernel_size1
    linear_index = block_indices

    offset_y = (0 * (0 >= (y_coord // 2)) + (y_coord // 2) * ((y_coord // 2) > (0)))
    offset_y_limit = (-1 + ((-1 + (kernel_size2 // 2)) * (((-1 + (kernel_size2 // 2)) <= (1 + (y_coord // 2))) + (1 + (y_coord // 2)) * ((1 + (y_coord // 2)) < (-1 + (kernel_size2 // 2))))) + ((-1 + (kernel_size2 // 2)) * (((-1 + (kernel_size2 // 2)) <= (1 + (y_coord // 2))) + (1 + (y_coord // 2)) * ((1 + (y_coord // 2)) < (-1 + (kernel_size2 // 2))))) * (((-1 + (kernel_size2 // 2)) * (((-1 + (kernel_size2 // 2)) <= (1 + (y_coord // 2))) + (1 + (y_coord // 2)) * ((1 + (y_coord // 2)) < (-1 + (kernel_size2 // 2))))) < (offset_y)))

    offset_x = (0 * (0 >= (x_coord // 2)) + (x_coord // 2) * ((x_coord // 2) > (0)))
    offset_x_limit = (-1 + ((-1 + (kernel_size2 // 2)) * (((-1 + (kernel_size2 // 2)) <= (1 + (x_coord // 2))) + (1 + (x_coord // 2)) * ((1 + (x_coord // 2)) < (-1 + (kernel_size2 // 2))))) + ((-1 + (kernel_size2 // 2)) * (((-1 + (kernel_size2 // 2)) <= (1 + (x_coord // 2))) + (1 + (x_coord // 2)) * ((1 + (x_coord // 2)) < (-1 + (kernel_size2 // 2))))) * (((-1 + (kernel_size2 // 2)) * (((-1 + (kernel_size2 // 2)) <= (1 + (x_coord // 2))) + (1 + (x_coord // 2)) * ((1 + (x_coord // 2)) < (-1 + (kernel_size2 // 2))))) < (offset_x)))

    input_index0 = (z_coord + (-1 * (offset_y))) + z_coord * (kernel_size2 // 2) * (kernel_size2 // 2) + (kernel_size2 // 2) * (offset_y) + (-2 * z_coord * (kernel_size2 // 2)) + offset_x
    input_index1 = (z_coord + (-1 * (offset_y))) + z_coord * (kernel_size2 // 2) * (kernel_size2 // 2) + (kernel_size2 // 2) * (offset_y) + (-2 * z_coord * (kernel_size2 // 2)) + offset_x

    input_value0 = tl.load(input_ptr0 + input_index0, valid_mask, eviction_policy='evict_last')
    input_value1 = tl.load(input_ptr1 + input_index1, valid_mask, eviction_policy='evict_last')

    divisor = tl.full([1], 2, tl.int32)
    quotient = tl.where((input_value0 < 0) != (divisor < 0), tl.where(input_value0 % divisor != 0, input_value0 // divisor - 1, input_value0 // divisor), input_value0 // divisor)
    product = quotient * divisor
    remainder = input_value0 - product

    offset_y_double = 2 * (offset_y * (offset_y <= offset_y_limit) + (-1 + offset_y_limit) * (offset_y_limit < offset_y))
    offset_y_double_plus_quotient = offset_y_double + quotient

    offset_x_double = 2 * (offset_x * (offset_x <= offset_x_limit) + (-1 + offset_x_limit) * (offset_x_limit < offset_x))
    offset_x_double_plus_remainder = offset_x_double + remainder

    kernel_size0 = kernel_size0
    linear_offset = offset_y_double_plus_quotient * kernel_size0
    final_index = linear_offset + offset_x_double_plus_remainder

    current_index = block_indices % kernel_size1
    match_condition = final_index == current_index

    zero_value = 0.0
    output_value = tl.where(match_condition, input_value1, zero_value)

    tl.store(output_ptr0 + linear_index, output_value, valid_mask)