# From: 13_ConvTranspose3d_Mean_Add_Softmax_Tanh_Scaling

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_convolution_backward_3(
    input_ptr, output_ptr, kernel_size_z, kernel_size_y, kernel_size_x, 
    input_num_elements, output_num_elements, XBLOCK: tl.constexpr, 
    RBLOCK: tl.constexpr
):
    input_num_elements = 336
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    output_base = tl.arange(0, RBLOCK)[None, :]
    input_x = (input_index % 21)
    input_y = input_index // 21
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_flat_index = input_index

    for output_offset in range(0, output_num_elements, RBLOCK):
        output_index = output_offset + output_base
        output_mask = output_index < output_num_elements
        output_flat_index = output_index

        temp_index = (
            output_flat_index + input_x * (
                triton_helpers.div_floor_integer(
                    20 + (-1) * kernel_size_x + 2 * kernel_size_x**2 + 
                    (-8) * kernel_size_y * kernel_size_x**2 + 
                    (-4) * kernel_size_x * kernel_size_y**2 + 
                    4 * kernel_size_x * kernel_size_y + 
                    8 * kernel_size_x**2 * kernel_size_y**2, 
                    21
                )
            )
        )

        temp_limit = (
            (-1) * kernel_size_x + 2 * kernel_size_x**2 + 
            (-8) * kernel_size_y * kernel_size_x**2 + 
            (-4) * kernel_size_x * kernel_size_y**2 + 
            4 * kernel_size_x * kernel_size_y + 
            8 * kernel_size_x**2 * kernel_size_y**2
        )

        temp_condition = temp_index < temp_limit

        input_value = tl.load(
            input_ptr + (
                (-1) * input_y + 
                (-1) * (
                    (output_flat_index + input_x * (
                        triton_helpers.div_floor_integer(
                            20 + (-1) * kernel_size_x + 2 * kernel_size_x**2 + 
                            (-8) * kernel_size_y * kernel_size_x**2 + 
                            (-4) * kernel_size_x * kernel_size_y**2 + 
                            4 * kernel_size_x * kernel_size_y + 
                            8 * kernel_size_x**2 * kernel_size_y**2, 
                            21
                        )
                    ) // (-1 + 2 * kernel_size_y)
                ) % (-1 + 2 * kernel_size_y)
            ) + 
            (-16) * (
                (output_flat_index + input_x * (
                    triton_helpers.div_floor_integer(
                        20 + (-1) * kernel_size_x + 2 * kernel_size_x**2 + 
                        (-8) * kernel_size_y * kernel_size_x**2 + 
                        (-4) * kernel_size_x * kernel_size_y**2 + 
                        4 * kernel_size_x * kernel_size_y + 
                        8 * kernel_size_x**2 * kernel_size_y**2, 
                        21
                    )
                ) // kernel_size_z
            ) % kernel_size_x
            ) + 
            (-64) * kernel_size_y**2 * (
                (output_flat_index + input_x * (
                    triton_helpers.div_floor_integer(
                        20 + (-1) * kernel_size_x + 2 * kernel_size_x**2 + 
                        (-8) * kernel_size_y * kernel_size_x**2 + 
                        (-4) * kernel_size_x * kernel_size_y**2 + 
                        4 * kernel_size_x * kernel_size_y + 
                        8 * kernel_size_x**2 * kernel_size_y**2, 
                        21
                    )
                ) // kernel_size_z
            ) % kernel_size_x
            ) + 
            (-4) * kernel_size_y * (
                (output_flat_index + input_x * (
                    triton_helpers.div_floor_integer(
                        20 + (-1) * kernel_size_x + 2 * kernel_size_x**2 + 
                        (-8) * kernel_size_y * kernel_size_x**2 + 
                        (-4) * kernel_size_x * kernel_size_y**2 + 
                        4 * kernel_size_x * kernel_size_y + 
                        8 * kernel_size_x**2 * kernel_size_y**2, 
                        21
                    )
                ) // (1 + (-4) * kernel_size_y + 4 * kernel_size_y**2)
            ) % (-1 + 2 * kernel_size_x)
            ) + 
            (-4) * input_y * kernel_size_y**2 + 
            2 * kernel_size_x * input_y + 
            2 * kernel_size_y * (
                (output_flat_index + input_x * (
                    triton_helpers.div_floor_integer(
                        20 + (-1) * kernel_size_x + 2 * kernel_size_x**2 + 
                        (-8) * kernel_size_y * kernel_size_x**2 + 
                        (-4) * kernel_size_x * kernel_size_y**2 + 
                        4 * kernel_size_x * kernel_size_y + 
                        8 * kernel_size_x**2 * kernel_size_y**2, 
                        21
                    )
                ) // (-1 + 2 * kernel_size_y)
            ) % (-1 + 2 * kernel_size_y)
            ) + 
            4 * kernel_size_y * input_y + 
            4 * kernel_size_y**2 * (
                (output_flat_index + input_x * (
                    triton_helpers.div_floor_integer(
                        20 + (-1) * kernel_size_x + 2 * kernel_size_x**2 + 
                        (-8) * kernel_size_y * kernel_size_x**2 + 
                        (-4) * kernel_size_x * kernel_size_y**2 + 
                        4 * kernel_size_x * kernel_size_y + 
                        8 * kernel_size_x**2 * kernel_size_y**2, 
                        21
                    )
                ) // (1 + (-4) * kernel_size_y + 4 * kernel_size_y**2)
            ) % (-1 + 2 * kernel_size_x)
            ) + 
            32 * kernel_size_x * (
                (output_flat_index + input_x * (
                    triton_helpers.div_floor_integer(
                        20 + (-1) * kernel_size_x + 2 * kernel_size_x**2 + 
                        (-8) * kernel_size_y * kernel_size_x**2 + 
                        (-4) * kernel_size_x * kernel_size_y**2 + 
                        4 * kernel_size_x * kernel_size_y + 
                        8 * kernel_size_x**2 * kernel_size_y**2, 
                        21
                    )
                ) // kernel_size_z
            ) % kernel_size_x
            ) + 
            64 * kernel_size_y * (
                (output_flat_index + input_x * (
                    triton_helpers.div_floor_integer(
                        20 + (-1) * kernel_size_x + 2 * kernel_size_x**2 + 
                        (-8) * kernel_size_y * kernel_size_x**2 + 
                        (-4) * kernel_size_x * kernel_size_y**2 + 
                        4 * kernel_size_x * kernel_size_y + 
                        8 * kernel_size_x**2 * kernel_size_y**2, 
                        21
                    )
                ) // kernel_size_z
            ) % kernel_size_x
            ) + 
            (-128) * kernel_size_x * kernel_size_y * (
                (output_flat_index + input_x * (
                    triton_helpers.div_floor_integer(
                        20 + (-1) * kernel_size_x + 2 * kernel_size_x**2 + 
                        (-8) * kernel_size_y * kernel_size_x**2 + 
                        (-4) * kernel_size_x * kernel_size_y**2 + 
                        4 * kernel_size_x * kernel_size_y + 
                        8 * kernel_size_x**2 * kernel_size_y**2, 
                        21
                    )
                ) // kernel_size_z
            ) % kernel_size_x
            ) + 
            (-8) * kernel_size_x * kernel_size_y * input_y + 
            8 * kernel_size_x * input_y * kernel_size_y**2 + 
            128 * kernel_size_x * kernel_size_y**2 * (
                (output_flat_index + input_x * (
                    triton_helpers.div_floor_integer(
                        20 + (-1) * kernel_size_x + 2 * kernel_size_x**2 + 
                        (-8) * kernel_size_y * kernel_size_x**2 + 
                        (-4) * kernel_size_x * kernel_size_y**2 + 
                        4 * kernel_size_x * kernel_size_y + 
                        8 * kernel_size_x**2 * kernel_size_y**2, 
                        21
                    )
                ) // kernel_size_z
            ) % kernel_size_x
            ) + 
            ((output_flat_index + input_x * (
                triton_helpers.div_floor_integer(
                    20 + (-1) * kernel_size_x + 2 * kernel_size_x**2 + 
                    (-8) * kernel_size_y * kernel_size_x**2 + 
                    (-4) * kernel_size_x * kernel_size_y**2 + 
                    4 * kernel_size_x * kernel_size_y + 
                    8 * kernel_size_x**2 * kernel_size_y**2, 
                    21
                )
            )) % (-1 + 2 * kernel_size_y)
            ) + 
            ((output_flat_index + input_x * (
                triton_helpers.div_floor_integer(
                    20 + (-1) * kernel_size_x + 2 * kernel_size_x**2 + 
                    (-8) * kernel_size_y * kernel_size_x**2 + 
                    (-4) * kernel_size_x * kernel_size_y**2 + 
                    4 * kernel_size_x * kernel_size_y + 
                    8 * kernel_size_x**2 * kernel_size_y**2, 
                    21
                )
            )) // (1 + (-4) * kernel_size_y + 4 * kernel_size_y**2) % (-1 + 2 * kernel_size_x)
        ), output_mask & temp_condition & input_mask, eviction_policy='evict_last', other=0.0
        )

        temp_broadcast = tl.broadcast_to(input_value, [XBLOCK, RBLOCK])
        temp_accumulated = temp_accumulator + temp_broadcast
        temp_accumulator = tl.where(output_mask & input_mask, temp_accumulated, temp_accumulator)

    temp_sum = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(output_ptr + (input_flat_index), temp_sum, input_mask)