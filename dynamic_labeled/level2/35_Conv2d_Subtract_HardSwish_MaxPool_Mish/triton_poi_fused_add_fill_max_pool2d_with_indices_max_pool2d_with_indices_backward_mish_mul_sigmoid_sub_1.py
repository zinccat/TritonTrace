# From: 35_Conv2d_Subtract_HardSwish_MaxPool_Mish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_fill_max_pool2d_with_indices_max_pool2d_with_indices_backward_mish_mul_sigmoid_sub_1(
    input_ptr0, input_ptr1, output_ptr0, kernel_size0, kernel_size1, kernel_size2, total_elements, BLOCK_SIZE : tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < total_elements

    x_coord = block_indices % kernel_size0
    y_coord = (block_indices // kernel_size0) % kernel_size0
    z_coord = block_indices // kernel_size1

    y_centered = (y_coord // 2)
    x_centered = (x_coord // 2)

    y_mask = (0 * (0 >= y_centered) + y_centered * (y_centered > 0))
    x_mask = (0 * (0 >= x_centered) + x_centered * (x_centered > 0))

    y_range = (-1 + ((-1 + (kernel_size2 // 2)) * ((-1 + (kernel_size2 // 2)) <= (1 + y_centered)) + (1 + y_centered) * ((1 + y_centered) < (-1 + (kernel_size2 // 2)))))
    x_range = (-1 + ((-1 + (kernel_size2 // 2)) * ((-1 + (kernel_size2 // 2)) <= (1 + x_centered)) + (1 + x_centered) * ((1 + x_centered) < (-1 + (kernel_size2 // 2)))))

    y_condition = (y_mask <= y_range) & (y_range < y_mask)
    x_condition = (x_mask <= x_range) & (x_range < x_mask)

    index_offset = (
        z_coord + 
        (-1 * (y_mask * (y_mask <= y_range) + y_range * y_condition)) * 
        (kernel_size2 // 2) * (kernel_size2 // 2) + 
        (kernel_size2 // 2) * (y_mask * (y_mask <= y_range) + y_range * y_condition) + 
        (-2 * z_coord * (kernel_size2 // 2)) + 
        (x_mask * (x_mask <= x_range) + x_range * x_condition)
    )

    input_value0 = tl.load(input_ptr0 + index_offset, valid_mask, eviction_policy='evict_last')
    input_value1 = tl.load(input_ptr1 + index_offset, valid_mask, eviction_policy='evict_last')

    divisor = tl.full([1], 2, tl.int32)
    quotient = tl.where((input_value0 < 0) != (divisor < 0), 
                        tl.where(input_value0 % divisor != 0, input_value0 // divisor - 1, input_value0 // divisor), 
                        input_value0 // divisor)
    product = quotient * divisor
    remainder = input_value0 - product

    y_stride = 2 * (y_mask * (y_mask <= y_range) + y_range * y_condition)
    y_stride_adjusted = y_stride + quotient

    x_stride = 2 * (x_mask * (x_mask <= x_range) + x_range * x_condition)
    x_stride_adjusted = x_stride + remainder

    kernel_stride = kernel_size0
    final_index = y_stride_adjusted * kernel_stride + x_stride_adjusted

    input_index = block_indices % kernel_size1
    match_condition = final_index == input_index

    output_value = tl.where(match_condition, input_value1, 0.0)
    tl.store(output_ptr0 + block_indices, output_value, valid_mask)