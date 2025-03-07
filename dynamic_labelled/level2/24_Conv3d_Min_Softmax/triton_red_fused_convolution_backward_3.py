# From: 24_Conv3d_Min_Softmax

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_convolution_backward_3(
    input_ptr, output_ptr, kernel_size_0, kernel_size_1, kernel_size_2, kernel_size_3, 
    input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    input_num_elements = 336
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_block_0 = input_index % 16
    temp_buffer = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_block_3 = input_index

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_block_2 = reduction_index

        temp_index_0 = (
            reduction_block_2 + input_index // 16 * (
                triton_helpers.div_floor_integer(
                    20 + (-8 * kernel_size_0) + (-2 * kernel_size_0 * kernel_size_2 * kernel_size_2) + 
                    4 * kernel_size_0 * kernel_size_1 + 8 * kernel_size_0 * kernel_size_2 + 
                    kernel_size_0 * kernel_size_1 * kernel_size_2 * kernel_size_2 + 
                    (-4 * kernel_size_0 * kernel_size_1 * kernel_size_2), 
                    21
                )
            )
        )

        temp_index_1 = (
            (-8 * kernel_size_0) + (-2 * kernel_size_0 * kernel_size_2 * kernel_size_2) + 
            4 * kernel_size_0 * kernel_size_1 + 8 * kernel_size_0 * kernel_size_2 + 
            kernel_size_0 * kernel_size_1 * kernel_size_2 * kernel_size_2 + 
            (-4 * kernel_size_0 * kernel_size_1 * kernel_size_2)
        )

        temp_index_2 = temp_index_0 < temp_index_1

        temp_index_3 = tl.load(
            input_ptr + (
                (-128 * (
                    ((reduction_block_2 + input_index // 16 * (
                        triton_helpers.div_floor_integer(
                            20 + (-8 * kernel_size_0) + (-2 * kernel_size_0 * kernel_size_2 * kernel_size_2) + 
                            4 * kernel_size_0 * kernel_size_1 + 8 * kernel_size_0 * kernel_size_2 + 
                            kernel_size_0 * kernel_size_1 * kernel_size_2 * kernel_size_2 + 
                            (-4 * kernel_size_0 * kernel_size_1 * kernel_size_2), 
                            21
                        )
                    ) // ((-8) + (-2) * kernel_size_2 * kernel_size_2 + 4 * kernel_size_1 + 8 * kernel_size_2 + 
                         kernel_size_1 * kernel_size_2 * kernel_size_2 + (-4 * kernel_size_1 * kernel_size_2))) % kernel_size_0)
                ) + (-8 * input_block_0) + 
                (-2 * (((reduction_block_2 + input_index // 16 * (
                    triton_helpers.div_floor_integer(
                        20 + (-8 * kernel_size_0) + (-2 * kernel_size_0 * kernel_size_2 * kernel_size_2) + 
                        4 * kernel_size_0 * kernel_size_1 + 8 * kernel_size_0 * kernel_size_2 + 
                        kernel_size_0 * kernel_size_1 * kernel_size_2 * kernel_size_2 + 
                        (-4 * kernel_size_0 * kernel_size_1 * kernel_size_2), 
                        21
                    )
                ) // ((-2) + kernel_size_2)) % ((-2) + kernel_size_2)))) + 
                4 * (((reduction_block_2 + input_index // 16 * (
                    triton_helpers.div_floor_integer(
                        20 + (-8 * kernel_size_0) + (-2 * kernel_size_0 * kernel_size_2 * kernel_size_2) + 
                        4 * kernel_size_0 * kernel_size_1 + 8 * kernel_size_0 * kernel_size_2 + 
                        kernel_size_0 * kernel_size_1 * kernel_size_2 * kernel_size_2 + 
                        (-4 * kernel_size_0 * kernel_size_1 * kernel_size_2), 
                        21
                    )
                ) // kernel_size_3) % ((-2) + kernel_size_1))) + 
                kernel_size_2 * (((reduction_block_2 + input_index // 16 * (
                    triton_helpers.div_floor_integer(
                        20 + (-8 * kernel_size_0) + (-2 * kernel_size_0 * kernel_size_2 * kernel_size_2) + 
                        4 * kernel_size_0 * kernel_size_1 + 8 * kernel_size_0 * kernel_size_2 + 
                        kernel_size_0 * kernel_size_1 * kernel_size_2 * kernel_size_2 + 
                        (-4 * kernel_size_0 * kernel_size_1 * kernel_size_2), 
                        21
                    )
                ) // ((-2) + kernel_size_2)) % ((-2) + kernel_size_2))) + 
                kernel_size_2 * kernel_size_2 * (((reduction_block_2 + input_index // 16 * (
                    triton_helpers.div_floor_integer(
                        20 + (-8 * kernel_size_0) + (-2 * kernel_size_0 * kernel_size_2 * kernel_size_2) + 
                        4 * kernel_size_0 * kernel_size_1 + 8 * kernel_size_0 * kernel_size_2 + 
                        kernel_size_0 * kernel_size_1 * kernel_size_2 * kernel_size_2 + 
                        (-4 * kernel_size_0 * kernel_size_1 * kernel_size_2), 
                        21
                    )
                ) // kernel_size_3) % ((-2) + kernel_size_1))) + 
                (-32 * kernel_size_2 * kernel_size_2 * (
                    ((reduction_block_2 + input_index // 16 * (
                        triton_helpers.div_floor_integer(
                            20 + (-8 * kernel_size_0) + (-2 * kernel_size_0 * kernel_size_2 * kernel_size_2) + 
                            4 * kernel_size_0 * kernel_size_1 + 8 * kernel_size_0 * kernel_size_2 + 
                            kernel_size_0 * kernel_size_1 * kernel_size_2 * kernel_size_2 + 
                            (-4 * kernel_size_0 * kernel_size_1 * kernel_size_2), 
                            21
                        )
                    ) // ((-8) + (-2) * kernel_size_2 * kernel_size_2 + 4 * kernel_size_1 + 8 * kernel_size_2 + 
                         kernel_size_1 * kernel_size_2 * kernel_size_2 + (-4 * kernel_size_1 * kernel_size_2))) % kernel_size_0)
                ) + 
                (-4 * kernel_size_2 * (
                    ((reduction_block_2 + input_index // 16 * (
                        triton_helpers.div_floor_integer(
                            20 + (-8 * kernel_size_0) + (-2 * kernel_size_0 * kernel_size_2 * kernel_size_2) + 
                            4 * kernel_size_0 * kernel_size_1 + 8 * kernel_size_0 * kernel_size_2 + 
                            kernel_size_0 * kernel_size_1 * kernel_size_2 * kernel_size_2 + 
                            (-4 * kernel_size_0 * kernel_size_1 * kernel_size_2), 
                            21
                        )
                    ) // kernel_size_3) % ((-2) + kernel_size_1))
                ) + 
                (-2 * input_block_0 * kernel_size_2 * kernel_size_2) + 
                4 * kernel_size_1 * input_block_0 + 
                8 * kernel_size_2 * input_block_0 + 
                64 * kernel_size_1 * (
                    ((reduction_block_2 + input_index // 16 * (
                        triton_helpers.div_floor_integer(
                            20 + (-8 * kernel_size_0) + (-2 * kernel_size_0 * kernel_size_2 * kernel_size_2) + 
                            4 * kernel_size_0 * kernel_size_1 + 8 * kernel_size_0 * kernel_size_2 + 
                            kernel_size_0 * kernel_size_1 * kernel_size_2 * kernel_size_2 + 
                            (-4 * kernel_size_0 * kernel_size_1 * kernel_size_2), 
                            21
                        )
                    ) // ((-8) + (-2) * kernel_size_2 * kernel_size_2 + 4 * kernel_size_1 + 8 * kernel_size_2 + 
                         kernel_size_1 * kernel_size_2 * kernel_size_2 + (-4 * kernel_size_1 * kernel_size_2))) % kernel_size_0)
                ) + 
                128 * kernel_size_2 * (
                    ((reduction_block_2 + input_index // 16 * (
                        triton_helpers.div_floor_integer(
                            20 + (-8 * kernel_size_0) + (-2 * kernel_size_0 * kernel_size_2 * kernel_size_2) + 
                            4 * kernel_size_0 * kernel_size_1 + 8 * kernel_size_0 * kernel_size_2 + 
                            kernel_size_0 * kernel_size_1 * kernel_size_2 * kernel_size_2 + 
                            (-4 * kernel_size_0 * kernel_size_1 * kernel_size_2), 
                            21
                        )
                    ) // ((-8) + (-2) * kernel_size_2 * kernel_size_2 + 4 * kernel_size_1 + 8 * kernel_size_2 + 
                         kernel_size_1 * kernel_size_2 * kernel_size_2 + (-4 * kernel_size_1 * kernel_size_2))) % kernel_size_0)
                ) + 
                kernel_size_1 * input_block_0 * kernel_size_2 * kernel_size_2 + 
                (-64 * kernel_size_1 * kernel_size_2 * (
                    ((reduction_block_2 + input_index // 16 * (
                        triton_helpers.div_floor_integer(
                            20 + (-8 * kernel_size_0) + (-2 * kernel_size_0 * kernel_size_2 * kernel_size_2) + 
                            4 * kernel_size_0 * kernel_size_1 + 8 * kernel_size_0 * kernel_size_2 + 
                            kernel_size_0 * kernel_size_1 * kernel_size_2 * kernel_size_2 + 
                            (-4 * kernel_size_0 * kernel_size_1 * kernel_size_2), 
                            21
                        )
                    ) // ((-8) + (-2) * kernel_size_2 * kernel_size_2 + 4 * kernel_size_1 + 8 * kernel_size_2 + 
                         kernel_size_1 * kernel_size_2 * kernel_size_2 + (-4 * kernel_size_1 * kernel_size_2))) % kernel_size_0)
                ) + 
                (-4 * kernel_size_1 * kernel_size_2 * input_block_0) + 
                16 * kernel_size_1 * kernel_size_2 * kernel_size_2 * (
                    ((reduction_block_2 + input_index // 16 * (
                        triton_helpers.div_floor_integer(
                            20 + (-8 * kernel_size_0) + (-2 * kernel_size_0 * kernel_size_2 * kernel_size_2) + 
                            4 * kernel_size_0 * kernel_size_1 + 8 * kernel_size_0 * kernel_size_2 + 
                            kernel_size_0 * kernel_size_1 * kernel_size_2 * kernel_size_2 + 
                            (-4 * kernel_size_0 * kernel_size_1 * kernel_size_2), 
                            21
                        )
                    ) // ((-8) + (-2) * kernel_size_2 * kernel_size_2 + 4 * kernel_size_1 + 8 * kernel_size_2 + 
                         kernel_size_1 * kernel_size_2 * kernel_size_2 + (-4 * kernel_size_1 * kernel_size_2))) % kernel_size_0)
                ) + 
                ((reduction_block_2 + input_index // 16 * (
                    triton_helpers.div_floor_integer(
                        20 + (-8 * kernel_size_0) + (-2 * kernel_size_0 * kernel_size_2 * kernel_size_2) + 
                        4 * kernel_size_0 * kernel_size_1 + 8 * kernel_size_0 * kernel_size_2 + 
                        kernel_size_0 * kernel_size_1 * kernel_size_2 * kernel_size_2 + 
                        (-4 * kernel_size_0 * kernel_size_1 * kernel_size_2), 
                        21
                    )
                )) % ((-2) + kernel_size_2))
            ), 
            reduction_mask & temp_index_2 & input_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )

        temp_index_4 = tl.broadcast_to(temp_index_3, [XBLOCK, RBLOCK])
        temp_index_6 = temp_buffer + temp_index_4
        temp_buffer = tl.where(reduction_mask & input_mask, temp_index_6, temp_buffer)

    temp_index_5 = tl.sum(temp_buffer, 1)[:, None]
    tl.store(output_ptr + (input_block_3), temp_index_5, input_mask)