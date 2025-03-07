# From: 47_Conv3d_Mish_Tanh

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_convolution_backward_1red_fused_convolution_backward_1(
    input_ptr, output_ptr, kernel_size_x, kernel_size_y, input_num_elements, output_num_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    input_num_elements = 400
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    output_base = tl.arange(0, RBLOCK)[None, :]
    input_x = (input_index % 25)
    input_y = input_index // 25
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_flat_index = input_index

    for output_offset in range(0, output_num_elements, RBLOCK):
        output_index = output_offset + output_base
        output_mask = output_index < output_num_elements
        output_flat_index = output_index

        temp_index = (
            output_flat_index + input_x * (
                triton_helpers.div_floor_integer(
                    24 + ((-8) * kernel_size_x) + 4 * kernel_size_x * kernel_size_x + 
                    kernel_size_x * kernel_size_x * kernel_size_y * kernel_size_y + 
                    ((-4) * kernel_size_y * kernel_size_x * kernel_size_x) + 
                    ((-2) * kernel_size_x * kernel_size_y * kernel_size_y) + 
                    8 * kernel_size_x * kernel_size_y, 25
                )
            )
        )

        temp_limit = (
            ((-8) * kernel_size_x) + 4 * kernel_size_x * kernel_size_x + 
            kernel_size_x * kernel_size_x * kernel_size_y * kernel_size_y + 
            ((-4) * kernel_size_y * kernel_size_x * kernel_size_x) + 
            ((-2) * kernel_size_x * kernel_size_y * kernel_size_y) + 
            8 * kernel_size_x * kernel_size_y
        )

        temp_condition = temp_index < temp_limit

        input_load_index = (
            (((-128) * (
                ((output_flat_index + input_x * (
                    triton_helpers.div_floor_integer(
                        24 + ((-8) * kernel_size_x) + 4 * kernel_size_x * kernel_size_x + 
                        kernel_size_x * kernel_size_x * kernel_size_y * kernel_size_y + 
                        ((-4) * kernel_size_y * kernel_size_x * kernel_size_x) + 
                        ((-2) * kernel_size_x * kernel_size_y * kernel_size_y) + 
                        8 * kernel_size_x * kernel_size_y, 25
                    )
                ) // ((-8) + ((-2) * kernel_size_y * kernel_size_y) + 4 * kernel_size_x + 
                8 * kernel_size_y + kernel_size_x * kernel_size_y * kernel_size_y + 
                ((-4) * kernel_size_x * kernel_size_y))) % kernel_size_x
            )) + ((-8) * input_y) + ((-2) * (
                ((output_flat_index + input_x * (
                    triton_helpers.div_floor_integer(
                        24 + ((-8) * kernel_size_x) + 4 * kernel_size_x * kernel_size_x + 
                        kernel_size_x * kernel_size_x * kernel_size_y * kernel_size_y + 
                        ((-4) * kernel_size_y * kernel_size_x * kernel_size_x) + 
                        ((-2) * kernel_size_x * kernel_size_y * kernel_size_y) + 
                        8 * kernel_size_x * kernel_size_y, 25
                    )
                ) // ((-2) + kernel_size_y)) % ((-2) + kernel_size_y)
            )) + 4 * (
                ((output_flat_index + input_x * (
                    triton_helpers.div_floor_integer(
                        24 + ((-8) * kernel_size_x) + 4 * kernel_size_x * kernel_size_x + 
                        kernel_size_x * kernel_size_x * kernel_size_y * kernel_size_y + 
                        ((-4) * kernel_size_y * kernel_size_x * kernel_size_x) + 
                        ((-2) * kernel_size_x * kernel_size_y * kernel_size_y) + 
                        8 * kernel_size_x * kernel_size_y, 25
                    )
                ) // (4 + kernel_size_y * kernel_size_y + ((-4) * kernel_size_y))) % ((-2) + kernel_size_x)
            ) + kernel_size_y * (
                ((output_flat_index + input_x * (
                    triton_helpers.div_floor_integer(
                        24 + ((-8) * kernel_size_x) + 4 * kernel_size_x * kernel_size_x + 
                        kernel_size_x * kernel_size_x * kernel_size_y * kernel_size_y + 
                        ((-4) * kernel_size_y * kernel_size_x * kernel_size_x) + 
                        ((-2) * kernel_size_x * kernel_size_y * kernel_size_y) + 
                        8 * kernel_size_x * kernel_size_y, 25
                    )
                ) // ((-2) + kernel_size_y)) % ((-2) + kernel_size_y)
            ) + kernel_size_y * kernel_size_y * (
                ((output_flat_index + input_x * (
                    triton_helpers.div_floor_integer(
                        24 + ((-8) * kernel_size_x) + 4 * kernel_size_x * kernel_size_x + 
                        kernel_size_x * kernel_size_x * kernel_size_y * kernel_size_y + 
                        ((-4) * kernel_size_y * kernel_size_x * kernel_size_x) + 
                        ((-2) * kernel_size_x * kernel_size_y * kernel_size_y) + 
                        8 * kernel_size_x * kernel_size_y, 25
                    )
                ) // (4 + kernel_size_y * kernel_size_y + ((-4) * kernel_size_y))) % ((-2) + kernel_size_x)
            ) + ((-32) * kernel_size_y * kernel_size_y * (
                ((output_flat_index + input_x * (
                    triton_helpers.div_floor_integer(
                        24 + ((-8) * kernel_size_x) + 4 * kernel_size_x * kernel_size_x + 
                        kernel_size_x * kernel_size_x * kernel_size_y * kernel_size_y + 
                        ((-4) * kernel_size_y * kernel_size_x * kernel_size_x) + 
                        ((-2) * kernel_size_x * kernel_size_y * kernel_size_y) + 
                        8 * kernel_size_x * kernel_size_y, 25
                    )
                ) // ((-8) + ((-2) * kernel_size_y * kernel_size_y) + 4 * kernel_size_x + 
                8 * kernel_size_y + kernel_size_x * kernel_size_y * kernel_size_y + 
                ((-4) * kernel_size_x * kernel_size_y))) % kernel_size_x
            )) + ((-4) * kernel_size_y * (
                ((output_flat_index + input_x * (
                    triton_helpers.div_floor_integer(
                        24 + ((-8) * kernel_size_x) + 4 * kernel_size_x * kernel_size_x + 
                        kernel_size_x * kernel_size_x * kernel_size_y * kernel_size_y + 
                        ((-4) * kernel_size_y * kernel_size_x * kernel_size_x) + 
                        ((-2) * kernel_size_x * kernel_size_y * kernel_size_y) + 
                        8 * kernel_size_x * kernel_size_y, 25
                    )
                ) // (4 + kernel_size_y * kernel_size_y + ((-4) * kernel_size_y))) % ((-2) + kernel_size_x)
            )) + ((-2) * input_y * kernel_size_y * kernel_size_y) + 4 * kernel_size_x * input_y + 
            8 * kernel_size_y * input_y + 64 * kernel_size_x * (
                ((output_flat_index + input_x * (
                    triton_helpers.div_floor_integer(
                        24 + ((-8) * kernel_size_x) + 4 * kernel_size_x * kernel_size_x + 
                        kernel_size_x * kernel_size_x * kernel_size_y * kernel_size_y + 
                        ((-4) * kernel_size_y * kernel_size_x * kernel_size_x) + 
                        ((-2) * kernel_size_x * kernel_size_y * kernel_size_y) + 
                        8 * kernel_size_x * kernel_size_y, 25
                    )
                ) // ((-8) + ((-2) * kernel_size_y * kernel_size_y) + 4 * kernel_size_x + 
                8 * kernel_size_y + kernel_size_x * kernel_size_y * kernel_size_y + 
                ((-4) * kernel_size_x * kernel_size_y))) % kernel_size_x
            )) + 128 * kernel_size_y * (
                ((output_flat_index + input_x * (
                    triton_helpers.div_floor_integer(
                        24 + ((-8) * kernel_size_x) + 4 * kernel_size_x * kernel_size_x + 
                        kernel_size_x * kernel_size_x * kernel_size_y * kernel_size_y + 
                        ((-4) * kernel_size_y * kernel_size_x * kernel_size_x) + 
                        ((-2) * kernel_size_x * kernel_size_y * kernel_size_y) + 
                        8 * kernel_size_x * kernel_size_y, 25
                    )
                ) // ((-8) + ((-2) * kernel_size_y * kernel_size_y) + 4 * kernel_size_x + 
                8 * kernel_size_y + kernel_size_x * kernel_size_y * kernel_size_y + 
                ((-4) * kernel_size_x * kernel_size_y))) % kernel_size_x
            )) + kernel_size_x * input_y * kernel_size_y * kernel_size_y + 
            ((-64) * kernel_size_x * kernel_size_y * (
                ((output_flat_index + input_x * (
                    triton_helpers.div_floor_integer(
                        24 + ((-8) * kernel_size_x) + 4 * kernel_size_x * kernel_size_x + 
                        kernel_size_x * kernel_size_x * kernel_size_y * kernel_size_y + 
                        ((-4) * kernel_size_y * kernel_size_x * kernel_size_x) + 
                        ((-2) * kernel_size_x * kernel_size_y * kernel_size_y) + 
                        8 * kernel_size_x * kernel_size_y, 25
                    )
                ) // ((-8) + ((-2) * kernel_size_y * kernel_size_y) + 4 * kernel_size_x + 
                8 * kernel_size_y + kernel_size_x * kernel_size_y * kernel_size_y + 
                ((-4) * kernel_size_x * kernel_size_y))) % kernel_size_x
            )) + ((-4) * kernel_size_x * kernel_size_y * input_y) + 
            16 * kernel_size_x * kernel_size_y * kernel_size_y * (
                ((output_flat_index + input_x * (
                    triton_helpers.div_floor_integer(
                        24 + ((-8) * kernel_size_x) + 4 * kernel_size_x * kernel_size_x + 
                        kernel_size_x * kernel_size_x * kernel_size_y * kernel_size_y + 
                        ((-4) * kernel_size_y * kernel_size_x * kernel_size_x) + 
                        ((-2) * kernel_size_x * kernel_size_y * kernel_size_y) + 
                        8 * kernel_size_x * kernel_size_y, 25
                    )
                ) // ((-8) + ((-2) * kernel_size_y * kernel_size_y) + 4 * kernel_size_x + 
                8 * kernel_size_y + kernel_size_x * kernel_size_y * kernel_size_y + 
                ((-4) * kernel_size_x * kernel_size_y))) % kernel_size_x
            )) + (
                (output_flat_index + input_x * (
                    triton_helpers.div_floor_integer(
                        24 + ((-8) * kernel_size_x) + 4 * kernel_size_x * kernel_size_x + 
                        kernel_size_x * kernel_size_x * kernel_size_y * kernel_size_y + 
                        ((-4) * kernel_size_y * kernel_size_x * kernel_size_x) + 
                        ((-2) * kernel_size_x * kernel_size_y * kernel_size_y) + 
                        8 * kernel_size_x * kernel_size_y, 25
                    )
                )) % ((-2) + kernel_size_y)
            )
        )

        temp_broadcast = tl.broadcast_to(temp_condition, [XBLOCK, RBLOCK])
        temp_accumulated = temp_accumulator + temp_broadcast
        temp_accumulator = tl.where(output_mask & temp_condition & input_mask, temp_accumulated, temp_accumulator)

    temp_result = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(output_ptr + (input_flat_index), temp_result, input_mask)