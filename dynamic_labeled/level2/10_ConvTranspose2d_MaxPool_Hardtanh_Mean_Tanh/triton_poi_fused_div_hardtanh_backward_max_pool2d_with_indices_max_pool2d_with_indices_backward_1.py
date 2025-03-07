# From: 10_ConvTranspose2d_MaxPool_Hardtanh_Mean_Tanh

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_div_hardtanh_backward_max_pool2d_with_indices_max_pool2d_with_indices_backward_1(
    input_ptr0, input_ptr1, output_ptr0, kernel_size_0, kernel_size_1, kernel_size_2, kernel_size_3, 
    num_elements, BLOCK_SIZE: tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_index = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = block_index < num_elements

    x_coord = block_index % kernel_size_0
    y_coord = (block_index // kernel_size_0) % kernel_size_0
    z_coord = block_index // kernel_size_1
    linear_index = block_index

    # Calculate indices for input_ptr0
    y_index = (0 if 0 >= (y_coord // 2) else y_coord // 2)
    y_limit = (-1 + (kernel_size_3 if kernel_size_3 <= (1 + (y_coord // 2)) else 1 + (y_coord // 2)))
    y_condition = (y_index <= y_limit) & (y_limit < y_index)

    x_index = (0 if 0 >= (x_coord // 2) else x_coord // 2)
    x_limit = (-1 + (kernel_size_3 if kernel_size_3 <= (1 + (x_coord // 2)) else 1 + (x_coord // 2)))
    x_condition = (x_index <= x_limit) & (x_limit < x_index)

    input_index_0 = (
        kernel_size_2 * z_coord + kernel_size_3 * (y_index * y_condition + y_limit * y_condition) + 
        x_index * x_condition + x_limit * x_condition
    )

    # Calculate indices for input_ptr1
    input_index_1 = (
        kernel_size_2 * z_coord + kernel_size_3 * (y_index * y_condition + y_limit * y_condition) + 
        x_index * x_condition + x_limit * x_condition
    )

    input_value_0 = tl.load(input_ptr0 + input_index_0, mask, eviction_policy='evict_last')
    input_value_1 = tl.load(input_ptr1 + input_index_1, mask, eviction_policy='evict_last')

    divisor = tl.full([1], 2, tl.int32)
    quotient = tl.where((input_value_0 < 0) != (divisor < 0), 
                        tl.where(input_value_0 % divisor != 0, input_value_0 // divisor - 1, input_value_0 // divisor), 
                        input_value_0 // divisor)

    product = quotient * divisor
    remainder = input_value_0 - product

    y_offset = 2 * (y_index * y_condition + y_limit * y_condition)
    y_adjusted = y_offset + quotient

    x_offset = 2 * (x_index * x_condition + x_limit * x_condition)
    x_adjusted = x_offset + remainder

    kernel_size = kernel_size_0
    final_index = y_adjusted * kernel_size + x_adjusted

    max_pool_index = block_index % kernel_size_1
    condition = final_index == max_pool_index

    output_value = tl.where(condition, input_value_1, 0.0)

    tl.store(output_ptr0 + linear_index, output_value, mask)