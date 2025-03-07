# From: 11_ConvTranspose2d_BatchNorm_Tanh_MaxPool_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_max_pool2d_with_indices_max_pool2d_with_indices_backward_4(
    input_ptr0, input_ptr1, output_ptr0, num_elements, BLOCK_SIZE: tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)

    x_col = block_indices % 64
    x_row = (block_indices // 64) % 64
    x_depth = block_indices // 4096
    x_linear_index = block_indices % 4096
    x_full_index = block_indices

    input_index_0 = (
        32 * (
            (0 if 0 >= (x_row // 2) else x_row // 2) *
            (0 if (0 if 0 >= (x_row // 2) else x_row // 2) > 0 else 1)
            <= (-1 + (32 if 32 <= (1 + (x_row // 2)) else 1 + (x_row // 2)))
            * (1 + (x_row // 2) if (1 + (x_row // 2)) < 32 else 0)
        ) + (-1 + (32 if 32 <= (1 + (x_row // 2)) else 1 + (x_row // 2)))
        * ((-1 + (32 if 32 <= (1 + (x_row // 2)) else 1 + (x_row // 2))) < (0 if 0 >= (x_row // 2) else x_row // 2))
    ) + 1024 * x_depth + (
        (0 if 0 >= (x_col // 2) else x_col // 2) *
        (0 if (0 if 0 >= (x_col // 2) else x_col // 2) > 0 else 1)
        <= (-1 + (32 if 32 <= (1 + (x_col // 2)) else 1 + (x_col // 2)))
        * (1 + (x_col // 2) if (1 + (x_col // 2)) < 32 else 0)
    ) + (-1 + (32 if 32 <= (1 + (x_col // 2)) else 1 + (x_col // 2)))
    * ((-1 + (32 if 32 <= (1 + (x_col // 2)) else 1 + (x_col // 2))) < (0 if 0 >= (x_col // 2) else x_col // 2))

    input_value_0 = tl.load(input_ptr0 + input_index_0, None, eviction_policy='evict_last')
    input_value_1 = tl.load(input_ptr1 + input_index_0, None, eviction_policy='evict_last')

    divisor = tl.full([1], 2, tl.int32)
    quotient = tl.where((input_value_0 < 0) != (divisor < 0), tl.where(input_value_0 % divisor != 0, input_value_0 // divisor - 1, input_value_0 // divisor), input_value_0 // divisor)
    product = quotient * divisor
    remainder = input_value_0 - product

    row_offset = 2 * (
        (0 if 0 >= (x_row // 2) else x_row // 2) *
        (0 if (0 if 0 >= (x_row // 2) else x_row // 2) > 0 else 1)
        <= (-1 + (32 if 32 <= (1 + (x_row // 2)) else 1 + (x_row // 2)))
        * (1 + (x_row // 2) if (1 + (x_row // 2)) < 32 else 0)
    ) + (-1 + (32 if 32 <= (1 + (x_row // 2)) else 1 + (x_row // 2)))
    * ((-1 + (32 if 32 <= (1 + (x_row // 2)) else 1 + (x_row // 2))) < (0 if 0 >= (x_row // 2) else x_row // 2))

    row_index = row_offset + quotient
    col_offset = 2 * (
        (0 if 0 >= (x_col // 2) else x_col // 2) *
        (0 if (0 if 0 >= (x_col // 2) else x_col // 2) > 0 else 1)
        <= (-1 + (32 if 32 <= (1 + (x_col // 2)) else 1 + (x_col // 2)))
        * (1 + (x_col // 2) if (1 + (x_col // 2)) < 32 else 0)
    ) + (-1 + (32 if 32 <= (1 + (x_col // 2)) else 1 + (x_col // 2)))
    * ((-1 + (32 if 32 <= (1 + (x_col // 2)) else 1 + (x_col // 2))) < (0 if 0 >= (x_col // 2) else x_col // 2))

    col_index = col_offset + remainder
    depth_multiplier = tl.full([1], 64, tl.int64)
    linear_index = row_index * depth_multiplier + col_index

    is_max = linear_index == x_linear_index
    output_value = tl.where(is_max, input_value_1, 0.0)

    tl.store(output_ptr0 + x_full_index, output_value, None)