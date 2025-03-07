# From: 58_ConvTranspose3d_LogSumExp_HardSwish_Subtract_Clamp_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_convolution_backward_6(
    input_ptr, output_ptr, kernel_size_z, kernel_size_y, kernel_size_x, num_kernels, input_num_elements, output_num_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    input_num_elements = 1968
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    output_base = tl.arange(0, RBLOCK)[None, :]
    input_channel = input_index % 16
    input_block = input_index // 16
    temp_result = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_linear_index = input_index

    for output_offset in range(0, output_num_elements, RBLOCK):
        output_index = output_offset + output_base
        output_mask = output_index < output_num_elements
        output_linear_index = output_index

        temp_index = output_linear_index + input_block * (
            triton_helpers.div_floor_integer(
                122 + (-1) * kernel_size_z + (-4) * kernel_size_z * kernel_size_x * kernel_size_x + 
                2 * kernel_size_z * kernel_size_y + 4 * kernel_size_z * kernel_size_x + 
                (-8) * kernel_size_z * kernel_size_y * kernel_size_x + 
                8 * kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x, 
                123
            )
        )

        temp_condition = temp_index < (
            (-1) * kernel_size_z + (-4) * kernel_size_z * kernel_size_x * kernel_size_x + 
            2 * kernel_size_z * kernel_size_y + 4 * kernel_size_z * kernel_size_x + 
            (-8) * kernel_size_z * kernel_size_y * kernel_size_x + 
            8 * kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x
        )

        temp_load_index = (
            (-1) * input_channel + 
            (-1) * (
                (output_linear_index + input_block * (
                    triton_helpers.div_floor_integer(
                        122 + (-1) * kernel_size_z + (-4) * kernel_size_z * kernel_size_x * kernel_size_x + 
                        2 * kernel_size_z * kernel_size_y + 4 * kernel_size_z * kernel_size_x + 
                        (-8) * kernel_size_z * kernel_size_y * kernel_size_x + 
                        8 * kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x, 
                        123
                    )
                ) // (-1 + 2 * kernel_size_x)) % (-1 + 2 * kernel_size_x)
            ) + 
            (-16) * (
                (output_linear_index + input_block * (
                    triton_helpers.div_floor_integer(
                        122 + (-1) * kernel_size_z + (-4) * kernel_size_z * kernel_size_x * kernel_size_x + 
                        2 * kernel_size_z * kernel_size_y + 4 * kernel_size_z * kernel_size_x + 
                        (-8) * kernel_size_z * kernel_size_y * kernel_size_x + 
                        8 * kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x, 
                        123
                    )
                ) // num_kernels) % kernel_size_z
            ) + 
            (-64) * kernel_size_x * kernel_size_x * (
                (output_linear_index + input_block * (
                    triton_helpers.div_floor_integer(
                        122 + (-1) * kernel_size_z + (-4) * kernel_size_z * kernel_size_x * kernel_size_x + 
                        2 * kernel_size_z * kernel_size_y + 4 * kernel_size_z * kernel_size_x + 
                        (-8) * kernel_size_z * kernel_size_y * kernel_size_x + 
                        8 * kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x, 
                        123
                    )
                ) // num_kernels) % kernel_size_z
            ) + 
            (-4) * kernel_size_x * (
                (output_linear_index + input_block * (
                    triton_helpers.div_floor_integer(
                        122 + (-1) * kernel_size_z + (-4) * kernel_size_z * kernel_size_x * kernel_size_x + 
                        2 * kernel_size_z * kernel_size_y + 4 * kernel_size_z * kernel_size_x + 
                        (-8) * kernel_size_z * kernel_size_y * kernel_size_x + 
                        8 * kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x, 
                        123
                    )
                ) // (1 + (-4) * kernel_size_x + 4 * kernel_size_x * kernel_size_x)) % (-1 + 2 * kernel_size_y)
            ) + 
            (-4) * input_channel * kernel_size_x * kernel_size_x + 
            2 * kernel_size_y * input_channel + 
            2 * kernel_size_x * (
                (output_linear_index + input_block * (
                    triton_helpers.div_floor_integer(
                        122 + (-1) * kernel_size_z + (-4) * kernel_size_z * kernel_size_x * kernel_size_x + 
                        2 * kernel_size_z * kernel_size_y + 4 * kernel_size_z * kernel_size_x + 
                        (-8) * kernel_size_z * kernel_size_y * kernel_size_x + 
                        8 * kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x, 
                        123
                    )
                ) // (-1 + 2 * kernel_size_x)) % (-1 + 2 * kernel_size_x)
            ) + 
            4 * kernel_size_x * input_channel + 
            4 * kernel_size_x * kernel_size_x * (
                (output_linear_index + input_block * (
                    triton_helpers.div_floor_integer(
                        122 + (-1) * kernel_size_z + (-4) * kernel_size_z * kernel_size_x * kernel_size_x + 
                        2 * kernel_size_z * kernel_size_y + 4 * kernel_size_z * kernel_size_x + 
                        (-8) * kernel_size_z * kernel_size_y * kernel_size_x + 
                        8 * kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x, 
                        123
                    )
                ) // (1 + (-4) * kernel_size_x + 4 * kernel_size_x * kernel_size_x)) % (-1 + 2 * kernel_size_y)
            ) + 
            32 * kernel_size_y * (
                (output_linear_index + input_block * (
                    triton_helpers.div_floor_integer(
                        122 + (-1) * kernel_size_z + (-4) * kernel_size_z * kernel_size_x * kernel_size_x + 
                        2 * kernel_size_z * kernel_size_y + 4 * kernel_size_z * kernel_size_x + 
                        (-8) * kernel_size_z * kernel_size_y * kernel_size_x + 
                        8 * kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x, 
                        123
                    )
                ) // num_kernels) % kernel_size_z
            ) + 
            64 * kernel_size_x * (
                (output_linear_index + input_block * (
                    triton_helpers.div_floor_integer(
                        122 + (-1) * kernel_size_z + (-4) * kernel_size_z * kernel_size_x * kernel_size_x + 
                        2 * kernel_size_z * kernel_size_y + 4 * kernel_size_z * kernel_size_x + 
                        (-8) * kernel_size_z * kernel_size_y * kernel_size_x + 
                        8 * kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x, 
                        123
                    )
                ) // num_kernels) % kernel_size_z
            ) + 
            (-128) * kernel_size_y * kernel_size_x * (
                (output_linear_index + input_block * (
                    triton_helpers.div_floor_integer(
                        122 + (-1) * kernel_size_z + (-4) * kernel_size_z * kernel_size_x * kernel_size_x + 
                        2 * kernel_size_z * kernel_size_y + 4 * kernel_size_z * kernel_size_x + 
                        (-8) * kernel_size_z * kernel_size_y * kernel_size_x + 
                        8 * kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x, 
                        123
                    )
                ) // num_kernels) % kernel_size_z
            ) + 
            (-8) * kernel_size_y * kernel_size_x * input_channel + 
            8 * kernel_size_y * input_channel * kernel_size_x * kernel_size_x + 
            128 * kernel_size_y * kernel_size_x * kernel_size_x * (
                (output_linear_index + input_block * (
                    triton_helpers.div_floor_integer(
                        122 + (-1) * kernel_size_z + (-4) * kernel_size_z * kernel_size_x * kernel_size_x + 
                        2 * kernel_size_z * kernel_size_y + 4 * kernel_size_z * kernel_size_x + 
                        (-8) * kernel_size_z * kernel_size_y * kernel_size_x + 
                        8 * kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x, 
                        123
                    )
                ) // num_kernels) % kernel_size_z
            ) + (
                (output_linear_index + input_block * (
                    triton_helpers.div_floor_integer(
                        122 + (-1) * kernel_size_z + (-4) * kernel_size_z * kernel_size_x * kernel_size_x + 
                        2 * kernel_size_z * kernel_size_y + 4 * kernel_size_z * kernel_size_x + 
                        (-8) * kernel_size_z * kernel_size_y * kernel_size_x + 
                        8 * kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x, 
                        123
                    )
                ) % (-1 + 2 * kernel_size_x)) + 
                (
                    (output_linear_index + input_block * (
                        triton_helpers.div_floor_integer(
                            122 + (-1) * kernel_size_z + (-4) * kernel_size_z * kernel_size_x * kernel_size_x + 
                            2 * kernel_size_z * kernel_size_y + 4 * kernel_size_z * kernel_size_x + 
                            (-8) * kernel_size_z * kernel_size_y * kernel_size_x + 
                            8 * kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x, 
                            123
                        )
                    ) // (1 + (-4) * kernel_size_x + 4 * kernel_size_x * kernel_size_x)) % (-1 + 2 * kernel_size_y)
                )
        )

        temp_data = tl.load(
            input_ptr + temp_load_index, 
            mask=(output_mask & temp_condition & input_mask), 
            eviction_policy='evict_last', 
            other=0.0
        )

        temp_broadcast = tl.broadcast_to(temp_data, [XBLOCK, RBLOCK])
        temp_accumulate = temp_result + temp_broadcast
        temp_result = tl.where(output_mask & input_mask, temp_accumulate, temp_result)

    temp_sum = tl.sum(temp_result, 1)[:, None]
    tl.store(output_ptr + input_linear_index, temp_sum, input_mask)