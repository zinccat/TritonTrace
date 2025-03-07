# From: 8_Conv3d_Divide_Max_GlobalAvgPool_BiasAdd_Sum

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_convolution_backward_3(in_ptr, out_ptr, kernel_size_z, kernel_size_y, kernel_size_x, input_num_elements, output_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    input_num_elements = 336
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    output_base = tl.arange(0, RBLOCK)[None, :]
    input_channel = input_index % 16
    input_block = input_index // 16
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_linear_index = input_index

    for output_offset in range(0, output_num_elements, RBLOCK):
        output_index = output_offset + output_base
        output_mask = output_index < output_num_elements
        output_linear_index = output_index

        temp_index = output_linear_index + input_block * (
            triton_helpers.div_floor_integer(
                20 + (-8 * kernel_size_z) + (-2 * kernel_size_z * kernel_size_z) + 4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + (-4 * kernel_size_z * kernel_size_y * kernel_size_x),
                21
            )
        )

        temp_condition = temp_index < (
            (-8 * kernel_size_z) + (-2 * kernel_size_z * kernel_size_z) + 4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + (-4 * kernel_size_z * kernel_size_y * kernel_size_x)
        )

        temp_load_index = (
            (-128) * (
                (((output_linear_index + input_block * (
                    triton_helpers.div_floor_integer(
                        20 + (-8 * kernel_size_z) + (-2 * kernel_size_z * kernel_size_z) + 4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + (-4 * kernel_size_z * kernel_size_y * kernel_size_x),
                        21
                    )
                ) // ((-8) + (-2) * kernel_size_x * kernel_size_x + 4 * kernel_size_y + 8 * kernel_size_x + kernel_size_y * kernel_size_x * kernel_size_x + (-4) * kernel_size_y * kernel_size_x)) % kernel_size_z)
            ) + (-8) * input_channel + (-2) * (
                (((output_linear_index + input_block * (
                    triton_helpers.div_floor_integer(
                        20 + (-8 * kernel_size_z) + (-2 * kernel_size_z * kernel_size_z) + 4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + (-4 * kernel_size_z * kernel_size_y * kernel_size_x),
                        21
                    )
                ) // ((-2) + kernel_size_x)) % ((-2) + kernel_size_x))
            ) + 4 * (
                (((output_linear_index + input_block * (
                    triton_helpers.div_floor_integer(
                        20 + (-8 * kernel_size_z) + (-2 * kernel_size_z * kernel_size_z) + 4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + (-4 * kernel_size_z * kernel_size_y * kernel_size_x),
                        21
                    )
                ) // (4 + kernel_size_x * kernel_size_x + (-4) * kernel_size_x)) % ((-2) + kernel_size_y))
            ) + kernel_size_x * (
                (((output_linear_index + input_block * (
                    triton_helpers.div_floor_integer(
                        20 + (-8 * kernel_size_z) + (-2 * kernel_size_z * kernel_size_z) + 4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + (-4 * kernel_size_z * kernel_size_y * kernel_size_x),
                        21
                    )
                ) // ((-2) + kernel_size_x)) % ((-2) + kernel_size_x))
            ) + kernel_size_x * kernel_size_x * (
                (((output_linear_index + input_block * (
                    triton_helpers.div_floor_integer(
                        20 + (-8 * kernel_size_z) + (-2 * kernel_size_z * kernel_size_z) + 4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + (-4 * kernel_size_z * kernel_size_y * kernel_size_x),
                        21
                    )
                ) // (4 + kernel_size_x * kernel_size_x + (-4) * kernel_size_x)) % ((-2) + kernel_size_y))
            ) + (-32) * kernel_size_x * kernel_size_x * (
                (((output_linear_index + input_block * (
                    triton_helpers.div_floor_integer(
                        20 + (-8 * kernel_size_z) + (-2 * kernel_size_z * kernel_size_z) + 4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + (-4 * kernel_size_z * kernel_size_y * kernel_size_x),
                        21
                    )
                ) // ((-8) + (-2) * kernel_size_x * kernel_size_x + 4 * kernel_size_y + 8 * kernel_size_x + kernel_size_y * kernel_size_x * kernel_size_x + (-4) * kernel_size_y * kernel_size_x)) % kernel_size_z)
            ) + (-4) * kernel_size_x * (
                (((output_linear_index + input_block * (
                    triton_helpers.div_floor_integer(
                        20 + (-8 * kernel_size_z) + (-2 * kernel_size_z * kernel_size_z) + 4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + (-4 * kernel_size_z * kernel_size_y * kernel_size_x),
                        21
                    )
                ) // (4 + kernel_size_x * kernel_size_x + (-4) * kernel_size_x)) % ((-2) + kernel_size_y))
            ) + (-2) * input_channel * kernel_size_x * kernel_size_x + 4 * kernel_size_y * input_channel + 8 * kernel_size_x * input_channel + 64 * kernel_size_y * (
                (((output_linear_index + input_block * (
                    triton_helpers.div_floor_integer(
                        20 + (-8 * kernel_size_z) + (-2 * kernel_size_z * kernel_size_z) + 4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + (-4 * kernel_size_z * kernel_size_y * kernel_size_x),
                        21
                    )
                ) // ((-8) + (-2) * kernel_size_x * kernel_size_x + 4 * kernel_size_y + 8 * kernel_size_x + kernel_size_y * kernel_size_x * kernel_size_x + (-4) * kernel_size_y * kernel_size_x)) % kernel_size_z)
            ) + 128 * kernel_size_x * (
                (((output_linear_index + input_block * (
                    triton_helpers.div_floor_integer(
                        20 + (-8 * kernel_size_z) + (-2 * kernel_size_z * kernel_size_z) + 4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + (-4 * kernel_size_z * kernel_size_y * kernel_size_x),
                        21
                    )
                ) // ((-8) + (-2) * kernel_size_x * kernel_size_x + 4 * kernel_size_y + 8 * kernel_size_x + kernel_size_y * kernel_size_x * kernel_size_x + (-4) * kernel_size_y * kernel_size_x)) % kernel_size_z)
            ) + kernel_size_y * input_channel * kernel_size_x * kernel_size_x + (-64) * kernel_size_y * kernel_size_x * (
                (((output_linear_index + input_block * (
                    triton_helpers.div_floor_integer(
                        20 + (-8 * kernel_size_z) + (-2 * kernel_size_z * kernel_size_z) + 4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + (-4 * kernel_size_z * kernel_size_y * kernel_size_x),
                        21
                    )
                ) // ((-8) + (-2) * kernel_size_x * kernel_size_x + 4 * kernel_size_y + 8 * kernel_size_x + kernel_size_y * kernel_size_x * kernel_size_x + (-4) * kernel_size_y * kernel_size_x)) % kernel_size_z)
            ) + (-4) * kernel_size_y * kernel_size_x * input_channel + 16 * kernel_size_y * kernel_size_x * kernel_size_x * (
                (((output_linear_index + input_block * (
                    triton_helpers.div_floor_integer(
                        20 + (-8 * kernel_size_z) + (-2 * kernel_size_z * kernel_size_z) + 4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + (-4 * kernel_size_z * kernel_size_y * kernel_size_x),
                        21
                    )
                ) // ((-8) + (-2) * kernel_size_x * kernel_size_x + 4 * kernel_size_y + 8 * kernel_size_x + kernel_size_y * kernel_size_x * kernel_size_x + (-4) * kernel_size_y * kernel_size_x)) % kernel_size_z)
            ) + (
                (output_linear_index + input_block * (
                    triton_helpers.div_floor_integer(
                        20 + (-8 * kernel_size_z) + (-2 * kernel_size_z * kernel_size_z) + 4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + (-4 * kernel_size_z * kernel_size_y * kernel_size_x),
                        21
                    )
                )) % ((-2) + kernel_size_x)
            )
        )

        temp_broadcast = tl.broadcast_to(temp_condition, [XBLOCK, RBLOCK])
        temp_accumulated = temp_accumulator + temp_broadcast
        temp_accumulator = tl.where(output_mask & input_mask, temp_accumulated, temp_accumulator)

    temp_sum = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(out_ptr + (input_linear_index), temp_sum, input_mask)