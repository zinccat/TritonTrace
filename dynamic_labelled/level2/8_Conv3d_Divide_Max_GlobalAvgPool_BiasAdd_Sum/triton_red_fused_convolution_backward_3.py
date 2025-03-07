# From: 8_Conv3d_Divide_Max_GlobalAvgPool_BiasAdd_Sum

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_convolution_backward_3(
    input_ptr, output_ptr, kernel_size_z, kernel_size_y, kernel_size_x, input_num_elements, output_num_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    input_num_elements = 336
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    output_base = tl.arange(0, RBLOCK)[None, :]
    input_1d = input_index // 16
    input_0d = (input_index % 16)
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_3d = input_index

    for output_offset in range(0, output_num_elements, RBLOCK):
        output_index = output_offset + output_base
        output_mask = output_index < output_num_elements
        output_2d = output_index

        temp_index = (
            output_2d + input_1d * (
                triton_helpers.div_floor_integer(
                    20 + ((-8) * kernel_size_z) + ((-2) * kernel_size_z * kernel_size_x * kernel_size_x) + 
                    4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + 
                    kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + 
                    ((-4) * kernel_size_z * kernel_size_y * kernel_size_x), 
                    21
                )
            )
        )

        temp_condition = (
            ((-8) * kernel_size_z) + ((-2) * kernel_size_z * kernel_size_x * kernel_size_x) + 
            4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + 
            kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + 
            ((-4) * kernel_size_z * kernel_size_y * kernel_size_x)
        )

        condition_mask = temp_index < temp_condition

        temp_load = tl.load(
            input_ptr + (
                ((-128) * (
                    (((output_2d + input_1d * (
                        triton_helpers.div_floor_integer(
                            20 + ((-8) * kernel_size_z) + ((-2) * kernel_size_z * kernel_size_x * kernel_size_x) + 
                            4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + 
                            kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + 
                            ((-4) * kernel_size_z * kernel_size_y * kernel_size_x), 
                            21
                        )
                    ) // ((-8) + ((-2) * kernel_size_x * kernel_size_x) + 4 * kernel_size_y + 8 * kernel_size_x + 
                    kernel_size_y * kernel_size_x * kernel_size_x + ((-4) * kernel_size_y * kernel_size_x))) % kernel_size_z)
                ) + ((-8) * input_0d) + ((-2) * (
                    (((output_2d + input_1d * (
                        triton_helpers.div_floor_integer(
                            20 + ((-8) * kernel_size_z) + ((-2) * kernel_size_z * kernel_size_x * kernel_size_x) + 
                            4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + 
                            kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + 
                            ((-4) * kernel_size_z * kernel_size_y * kernel_size_x), 
                            21
                        )
                    ) // ((-2) + kernel_size_x)) % ((-2) + kernel_size_x))
                ) + 4 * (
                    (((output_2d + input_1d * (
                        triton_helpers.div_floor_integer(
                            20 + ((-8) * kernel_size_z) + ((-2) * kernel_size_z * kernel_size_x * kernel_size_x) + 
                            4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + 
                            kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + 
                            ((-4) * kernel_size_z * kernel_size_y * kernel_size_x), 
                            21
                        )
                    ) // (4 + kernel_size_x * kernel_size_x + ((-4) * kernel_size_x))) % ((-2) + kernel_size_y))
                ) + kernel_size_x * (
                    (((output_2d + input_1d * (
                        triton_helpers.div_floor_integer(
                            20 + ((-8) * kernel_size_z) + ((-2) * kernel_size_z * kernel_size_x * kernel_size_x) + 
                            4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + 
                            kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + 
                            ((-4) * kernel_size_z * kernel_size_y * kernel_size_x), 
                            21
                        )
                    ) // ((-2) + kernel_size_x)) % ((-2) + kernel_size_x))
                ) + kernel_size_x * kernel_size_x * (
                    (((output_2d + input_1d * (
                        triton_helpers.div_floor_integer(
                            20 + ((-8) * kernel_size_z) + ((-2) * kernel_size_z * kernel_size_x * kernel_size_x) + 
                            4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + 
                            kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + 
                            ((-4) * kernel_size_z * kernel_size_y * kernel_size_x), 
                            21
                        )
                    ) // (4 + kernel_size_x * kernel_size_x + ((-4) * kernel_size_x))) % ((-2) + kernel_size_y))
                ) + ((-32) * kernel_size_x * kernel_size_x * (
                    (((output_2d + input_1d * (
                        triton_helpers.div_floor_integer(
                            20 + ((-8) * kernel_size_z) + ((-2) * kernel_size_z * kernel_size_x * kernel_size_x) + 
                            4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + 
                            kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + 
                            ((-4) * kernel_size_z * kernel_size_y * kernel_size_x), 
                            21
                        )
                    ) // ((-8) + ((-2) * kernel_size_x * kernel_size_x) + 4 * kernel_size_y + 8 * kernel_size_x + 
                    kernel_size_y * kernel_size_x * kernel_size_x + ((-4) * kernel_size_y * kernel_size_x))) % kernel_size_z)
                ) + ((-4) * kernel_size_x * (
                    (((output_2d + input_1d * (
                        triton_helpers.div_floor_integer(
                            20 + ((-8) * kernel_size_z) + ((-2) * kernel_size_z * kernel_size_x * kernel_size_x) + 
                            4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + 
                            kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + 
                            ((-4) * kernel_size_z * kernel_size_y * kernel_size_x), 
                            21
                        )
                    ) // (4 + kernel_size_x * kernel_size_x + ((-4) * kernel_size_x))) % ((-2) + kernel_size_y)))
                ) + ((-2) * input_0d * kernel_size_x * kernel_size_x) + 4 * kernel_size_y * input_0d + 
                8 * kernel_size_x * input_0d + 64 * kernel_size_y * (
                    (((output_2d + input_1d * (
                        triton_helpers.div_floor_integer(
                            20 + ((-8) * kernel_size_z) + ((-2) * kernel_size_z * kernel_size_x * kernel_size_x) + 
                            4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + 
                            kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + 
                            ((-4) * kernel_size_z * kernel_size_y * kernel_size_x), 
                            21
                        )
                    ) // ((-8) + ((-2) * kernel_size_x * kernel_size_x) + 4 * kernel_size_y + 8 * kernel_size_x + 
                    kernel_size_y * kernel_size_x * kernel_size_x + ((-4) * kernel_size_y * kernel_size_x))) % kernel_size_z)
                ) + 128 * kernel_size_x * (
                    (((output_2d + input_1d * (
                        triton_helpers.div_floor_integer(
                            20 + ((-8) * kernel_size_z) + ((-2) * kernel_size_z * kernel_size_x * kernel_size_x) + 
                            4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + 
                            kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + 
                            ((-4) * kernel_size_z * kernel_size_y * kernel_size_x), 
                            21
                        )
                    ) // ((-8) + ((-2) * kernel_size_x * kernel_size_x) + 4 * kernel_size_y + 8 * kernel_size_x + 
                    kernel_size_y * kernel_size_x * kernel_size_x + ((-4) * kernel_size_y * kernel_size_x))) % kernel_size_z)
                ) + kernel_size_y * input_0d * kernel_size_x * kernel_size_x + 
                ((-64) * kernel_size_y * kernel_size_x * (
                    (((output_2d + input_1d * (
                        triton_helpers.div_floor_integer(
                            20 + ((-8) * kernel_size_z) + ((-2) * kernel_size_z * kernel_size_x * kernel_size_x) + 
                            4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + 
                            kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + 
                            ((-4) * kernel_size_z * kernel_size_y * kernel_size_x), 
                            21
                        )
                    ) // ((-8) + ((-2) * kernel_size_x * kernel_size_x) + 4 * kernel_size_y + 8 * kernel_size_x + 
                    kernel_size_y * kernel_size_x * kernel_size_x + ((-4) * kernel_size_y * kernel_size_x))) % kernel_size_z)
                ) + ((-4) * kernel_size_y * kernel_size_x * input_0d) + 
                16 * kernel_size_y * kernel_size_x * kernel_size_x * (
                    (((output_2d + input_1d * (
                        triton_helpers.div_floor_integer(
                            20 + ((-8) * kernel_size_z) + ((-2) * kernel_size_z * kernel_size_x * kernel_size_x) + 
                            4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + 
                            kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + 
                            ((-4) * kernel_size_z * kernel_size_y * kernel_size_x), 
                            21
                        )
                    ) // ((-8) + ((-2) * kernel_size_x * kernel_size_x) + 4 * kernel_size_y + 8 * kernel_size_x + 
                    kernel_size_y * kernel_size_x * kernel_size_x + ((-4) * kernel_size_y * kernel_size_x))) % kernel_size_z)
                ) + (
                    (output_2d + input_1d * (
                        triton_helpers.div_floor_integer(
                            20 + ((-8) * kernel_size_z) + ((-2) * kernel_size_z * kernel_size_x * kernel_size_x) + 
                            4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + 
                            kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + 
                            ((-4) * kernel_size_z * kernel_size_y * kernel_size_x), 
                            21
                        )
                    )) % ((-2) + kernel_size_x)
                )
            ), 
            eviction_policy='evict_last', 
            other=0.0
        )

        temp_broadcast = tl.broadcast_to(condition_mask, [XBLOCK, RBLOCK])
        temp_accumulate = temp_accumulator + temp_broadcast
        temp_accumulator = tl.where(output_mask & input_mask, temp_accumulate, temp_accumulator)

    temp_sum = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(output_ptr + (input_3d), temp_sum, input_mask)