# From: 27_Conv3d_HardSwish_ReLU_Softmax_Mean

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_convolution_backward_3(
    input_ptr, output_ptr, kernel_size_z, kernel_size_y, kernel_size_x, kernel_size_w, input_num_elements, reduction_num_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    input_num_elements = 336
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_channel = input_index % 16
    input_block = input_index // 16
    temp_result = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_linear_index = input_index

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_linear_index = reduction_index

        temp_index = reduction_linear_index + input_block * (
            triton_helpers.div_floor_integer(
                20 + (-8 * kernel_size_z) + (-2 * kernel_size_z * kernel_size_x * kernel_size_x) + 
                4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + 
                kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + 
                (-4 * kernel_size_z * kernel_size_y * kernel_size_x), 
                21
            )
        )

        temp_condition = (
            (-8 * kernel_size_z) + (-2 * kernel_size_z * kernel_size_x * kernel_size_x) + 
            4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + 
            kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + 
            (-4 * kernel_size_z * kernel_size_y * kernel_size_x)
        )

        condition_mask = temp_index < temp_condition

        input_address = (
            (-128) * (
                (((reduction_linear_index + input_block * (
                    triton_helpers.div_floor_integer(
                        20 + (-8 * kernel_size_z) + (-2 * kernel_size_z * kernel_size_x * kernel_size_x) + 
                        4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + 
                        kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + 
                        (-4 * kernel_size_z * kernel_size_y * kernel_size_x), 
                        21
                    )
                ) // kernel_size_w) % kernel_size_z)
                + (-8) * input_channel
                + (-2) * (
                    ((reduction_linear_index + input_block * (
                        triton_helpers.div_floor_integer(
                            20 + (-8 * kernel_size_z) + (-2 * kernel_size_z * kernel_size_x * kernel_size_x) + 
                            4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + 
                            kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + 
                            (-4 * kernel_size_z * kernel_size_y * kernel_size_x), 
                            21
                        )
                    ) // ((-2) + kernel_size_x)) % ((-2) + kernel_size_x))
                )
                + 4 * (
                    (((reduction_linear_index + input_block * (
                        triton_helpers.div_floor_integer(
                            20 + (-8 * kernel_size_z) + (-2 * kernel_size_z * kernel_size_x * kernel_size_x) + 
                            4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + 
                            kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + 
                            (-4 * kernel_size_z * kernel_size_y * kernel_size_x), 
                            21
                        )
                    ) // (4 + kernel_size_x * kernel_size_x + (-4) * kernel_size_x)) % ((-2) + kernel_size_y))
                )
                + kernel_size_x * (
                    (((reduction_linear_index + input_block * (
                        triton_helpers.div_floor_integer(
                            20 + (-8 * kernel_size_z) + (-2 * kernel_size_z * kernel_size_x * kernel_size_x) + 
                            4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + 
                            kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + 
                            (-4 * kernel_size_z * kernel_size_y * kernel_size_x), 
                            21
                        )
                    ) // ((-2) + kernel_size_x)) % ((-2) + kernel_size_x))
                )
                + kernel_size_x * kernel_size_x * (
                    (((reduction_linear_index + input_block * (
                        triton_helpers.div_floor_integer(
                            20 + (-8 * kernel_size_z) + (-2 * kernel_size_z * kernel_size_x * kernel_size_x) + 
                            4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + 
                            kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + 
                            (-4 * kernel_size_z * kernel_size_y * kernel_size_x), 
                            21
                        )
                    ) // (4 + kernel_size_x * kernel_size_x + (-4) * kernel_size_x)) % ((-2) + kernel_size_y))
                )
                + (-32) * kernel_size_x * kernel_size_x * (
                    (((reduction_linear_index + input_block * (
                        triton_helpers.div_floor_integer(
                            20 + (-8 * kernel_size_z) + (-2 * kernel_size_z * kernel_size_x * kernel_size_x) + 
                            4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + 
                            kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + 
                            (-4 * kernel_size_z * kernel_size_y * kernel_size_x), 
                            21
                        )
                    ) // kernel_size_w) % kernel_size_z)
                )
                + (-4) * kernel_size_x * (
                    (((reduction_linear_index + input_block * (
                        triton_helpers.div_floor_integer(
                            20 + (-8 * kernel_size_z) + (-2 * kernel_size_z * kernel_size_x * kernel_size_x) + 
                            4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + 
                            kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + 
                            (-4 * kernel_size_z * kernel_size_y * kernel_size_x), 
                            21
                        )
                    ) // (4 + kernel_size_x * kernel_size_x + (-4) * kernel_size_x)) % ((-2) + kernel_size_y))
                )
                + (-2) * input_channel * kernel_size_x * kernel_size_x
                + 4 * kernel_size_y * input_channel
                + 8 * kernel_size_x * input_channel
                + 64 * kernel_size_y * (
                    (((reduction_linear_index + input_block * (
                        triton_helpers.div_floor_integer(
                            20 + (-8 * kernel_size_z) + (-2 * kernel_size_z * kernel_size_x * kernel_size_x) + 
                            4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + 
                            kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + 
                            (-4 * kernel_size_z * kernel_size_y * kernel_size_x), 
                            21
                        )
                    ) // kernel_size_w) % kernel_size_z))
                )
                + 128 * kernel_size_x * (
                    (((reduction_linear_index + input_block * (
                        triton_helpers.div_floor_integer(
                            20 + (-8 * kernel_size_z) + (-2 * kernel_size_z * kernel_size_x * kernel_size_x) + 
                            4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + 
                            kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + 
                            (-4 * kernel_size_z * kernel_size_y * kernel_size_x), 
                            21
                        )
                    ) // kernel_size_w) % kernel_size_z))
                )
                + kernel_size_y * input_channel * kernel_size_x * kernel_size_x
                + (-64) * kernel_size_y * kernel_size_x * (
                    (((reduction_linear_index + input_block * (
                        triton_helpers.div_floor_integer(
                            20 + (-8 * kernel_size_z) + (-2 * kernel_size_z * kernel_size_x * kernel_size_x) + 
                            4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + 
                            kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + 
                            (-4 * kernel_size_z * kernel_size_y * kernel_size_x), 
                            21
                        )
                    ) // kernel_size_w) % kernel_size_z))
                )
                + (-4) * kernel_size_y * kernel_size_x * input_channel
                + 16 * kernel_size_y * kernel_size_x * kernel_size_x * (
                    (((reduction_linear_index + input_block * (
                        triton_helpers.div_floor_integer(
                            20 + (-8 * kernel_size_z) + (-2 * kernel_size_z * kernel_size_x * kernel_size_x) + 
                            4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + 
                            kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + 
                            (-4 * kernel_size_z * kernel_size_y * kernel_size_x), 
                            21
                        )
                    ) // kernel_size_w) % kernel_size_z))
                )
                + (((reduction_linear_index + input_block * (
                    triton_helpers.div_floor_integer(
                        20 + (-8 * kernel_size_z) + (-2 * kernel_size_z * kernel_size_x * kernel_size_x) + 
                        4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + 
                        kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + 
                        (-4 * kernel_size_z * kernel_size_y * kernel_size_x), 
                        21
                    )
                ) % ((-2) + kernel_size_x))))
        )

        temp_data = tl.load(
            input_ptr + input_address, 
            mask=reduction_mask & condition_mask & input_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )

        broadcast_temp_data = tl.broadcast_to(temp_data, [XBLOCK, RBLOCK])
        temp_result += broadcast_temp_data
        temp_result = tl.where(reduction_mask & input_mask, temp_result, temp_result)

    result_sum = tl.sum(temp_result, 1)[:, None]
    tl.store(output_ptr + input_linear_index, result_sum, input_mask)