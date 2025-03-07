# From: 90_Conv3d_LeakyReLU_Sum_Clamp_GELU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_sum_2red_fused_sum_2(
    input_ptr, output_ptr, kernel_size_0, kernel_size_1, kernel_size_2, 
    input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, 
    RBLOCK: tl.constexpr
):
    input_num_elements = 336
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_1d = input_index // 16
    input_0d = (input_index % 16)
    temp_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_3d = input_index

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_2d = reduction_index

        temp_index_0 = reduction_2d + input_1d * (
            triton_helpers.div_floor_integer(
                20 + ((-8) * kernel_size_0) + ((-2) * kernel_size_0 * kernel_size_2 * kernel_size_2) + 
                4 * kernel_size_0 * kernel_size_1 + 8 * kernel_size_0 * kernel_size_2 + 
                kernel_size_0 * kernel_size_1 * kernel_size_2 * kernel_size_2 + 
                ((-4) * kernel_size_0 * kernel_size_1 * kernel_size_2), 
                21
            )
        )

        temp_index_1 = ((-8) * kernel_size_0) + ((-2) * kernel_size_0 * kernel_size_2 * kernel_size_2) + \
                       4 * kernel_size_0 * kernel_size_1 + 8 * kernel_size_0 * kernel_size_2 + \
                       kernel_size_0 * kernel_size_1 * kernel_size_2 * kernel_size_2 + \
                       ((-4) * kernel_size_0 * kernel_size_1 * kernel_size_2)

        temp_mask = temp_index_0 < temp_index_1

        temp_load = tl.load(
            input_ptr + (
                ((-128) * (
                    (((reduction_2d + input_1d * (
                        triton_helpers.div_floor_integer(
                            20 + ((-8) * kernel_size_0) + ((-2) * kernel_size_0 * kernel_size_2 * kernel_size_2) + 
                            4 * kernel_size_0 * kernel_size_1 + 8 * kernel_size_0 * kernel_size_2 + 
                            kernel_size_0 * kernel_size_1 * kernel_size_2 * kernel_size_2 + 
                            ((-4) * kernel_size_0 * kernel_size_1 * kernel_size_2), 
                            21
                        )
                    )) // ((-8) + ((-2) * kernel_size_2 * kernel_size_2) + 4 * kernel_size_1 + 8 * kernel_size_2 + 
                           kernel_size_1 * kernel_size_2 * kernel_size_2 + ((-4) * kernel_size_1 * kernel_size_2))) % kernel_size_0)
                ) + ((-8) * input_0d) + ((-2) * (
                    (((reduction_2d + input_1d * (
                        triton_helpers.div_floor_integer(
                            20 + ((-8) * kernel_size_0) + ((-2) * kernel_size_0 * kernel_size_2 * kernel_size_2) + 
                            4 * kernel_size_0 * kernel_size_1 + 8 * kernel_size_0 * kernel_size_2 + 
                            kernel_size_0 * kernel_size_1 * kernel_size_2 * kernel_size_2 + 
                            ((-4) * kernel_size_0 * kernel_size_1 * kernel_size_2), 
                            21
                        )
                    )) // ((-2) + kernel_size_2)) % ((-2) + kernel_size_2))
                )) + 4 * (
                    (((reduction_2d + input_1d * (
                        triton_helpers.div_floor_integer(
                            20 + ((-8) * kernel_size_0) + ((-2) * kernel_size_0 * kernel_size_2 * kernel_size_2) + 
                            4 * kernel_size_0 * kernel_size_1 + 8 * kernel_size_0 * kernel_size_2 + 
                            kernel_size_0 * kernel_size_1 * kernel_size_2 * kernel_size_2 + 
                            ((-4) * kernel_size_0 * kernel_size_1 * kernel_size_2), 
                            21
                        )
                    )) // (4 + kernel_size_2 * kernel_size_2 + ((-4) * kernel_size_2))) % ((-2) + kernel_size_1))
                ) + kernel_size_2 * (
                    (((reduction_2d + input_1d * (
                        triton_helpers.div_floor_integer(
                            20 + ((-8) * kernel_size_0) + ((-2) * kernel_size_0 * kernel_size_2 * kernel_size_2) + 
                            4 * kernel_size_0 * kernel_size_1 + 8 * kernel_size_0 * kernel_size_2 + 
                            kernel_size_0 * kernel_size_1 * kernel_size_2 * kernel_size_2 + 
                            ((-4) * kernel_size_0 * kernel_size_1 * kernel_size_2), 
                            21
                        )
                    )) // ((-2) + kernel_size_2)) % ((-2) + kernel_size_2))
                ) + kernel_size_2 * kernel_size_2 * (
                    (((reduction_2d + input_1d * (
                        triton_helpers.div_floor_integer(
                            20 + ((-8) * kernel_size_0) + ((-2) * kernel_size_0 * kernel_size_2 * kernel_size_2) + 
                            4 * kernel_size_0 * kernel_size_1 + 8 * kernel_size_0 * kernel_size_2 + 
                            kernel_size_0 * kernel_size_1 * kernel_size_2 * kernel_size_2 + 
                            ((-4) * kernel_size_0 * kernel_size_1 * kernel_size_2), 
                            21
                        )
                    )) // (4 + kernel_size_2 * kernel_size_2 + ((-4) * kernel_size_2))) % ((-2) + kernel_size_1))
                ) + ((-32) * kernel_size_2 * kernel_size_2 * (
                    (((reduction_2d + input_1d * (
                        triton_helpers.div_floor_integer(
                            20 + ((-8) * kernel_size_0) + ((-2) * kernel_size_0 * kernel_size_2 * kernel_size_2) + 
                            4 * kernel_size_0 * kernel_size_1 + 8 * kernel_size_0 * kernel_size_2 + 
                            kernel_size_0 * kernel_size_1 * kernel_size_2 * kernel_size_2 + 
                            ((-4) * kernel_size_0 * kernel_size_1 * kernel_size_2), 
                            21
                        )
                    )) // ((-8) + ((-2) * kernel_size_2 * kernel_size_2) + 4 * kernel_size_1 + 8 * kernel_size_2 + 
                           kernel_size_1 * kernel_size_2 * kernel_size_2 + ((-4) * kernel_size_1 * kernel_size_2))) % kernel_size_0)
                ) + ((-4) * kernel_size_2 * (
                    (((reduction_2d + input_1d * (
                        triton_helpers.div_floor_integer(
                            20 + ((-8) * kernel_size_0) + ((-2) * kernel_size_0 * kernel_size_2 * kernel_size_2) + 
                            4 * kernel_size_0 * kernel_size_1 + 8 * kernel_size_0 * kernel_size_2 + 
                            kernel_size_0 * kernel_size_1 * kernel_size_2 * kernel_size_2 + 
                            ((-4) * kernel_size_0 * kernel_size_1 * kernel_size_2), 
                            21
                        )
                    )) // (4 + kernel_size_2 * kernel_size_2 + ((-4) * kernel_size_2))) % ((-2) + kernel_size_1))
                )) + ((-2) * input_0d * kernel_size_2 * kernel_size_2) + 4 * kernel_size_1 * input_0d + \
                8 * kernel_size_2 * input_0d + 64 * kernel_size_1 * (
                    (((reduction_2d + input_1d * (
                        triton_helpers.div_floor_integer(
                            20 + ((-8) * kernel_size_0) + ((-2) * kernel_size_0 * kernel_size_2 * kernel_size_2) + 
                            4 * kernel_size_0 * kernel_size_1 + 8 * kernel_size_0 * kernel_size_2 + 
                            kernel_size_0 * kernel_size_1 * kernel_size_2 * kernel_size_2 + 
                            ((-4) * kernel_size_0 * kernel_size_1 * kernel_size_2), 
                            21
                        )
                    )) // ((-8) + ((-2) * kernel_size_2 * kernel_size_2) + 4 * kernel_size_1 + 8 * kernel_size_2 + 
                           kernel_size_1 * kernel_size_2 * kernel_size_2 + ((-4) * kernel_size_1 * kernel_size_2))) % kernel_size_0)
                ) + 128 * kernel_size_2 * (
                    (((reduction_2d + input_1d * (
                        triton_helpers.div_floor_integer(
                            20 + ((-8) * kernel_size_0) + ((-2) * kernel_size_0 * kernel_size_2 * kernel_size_2) + 
                            4 * kernel_size_0 * kernel_size_1 + 8 * kernel_size_0 * kernel_size_2 + 
                            kernel_size_0 * kernel_size_1 * kernel_size_2 * kernel_size_2 + 
                            ((-4) * kernel_size_0 * kernel_size_1 * kernel_size_2), 
                            21
                        )
                    )) // ((-8) + ((-2) * kernel_size_2 * kernel_size_2) + 4 * kernel_size_1 + 8 * kernel_size_2 + 
                           kernel_size_1 * kernel_size_2 * kernel_size_2 + ((-4) * kernel_size_1 * kernel_size_2))) % kernel_size_0)
                ) + kernel_size_1 * input_0d * kernel_size_2 * kernel_size_2 + \
                ((-64) * kernel_size_1 * kernel_size_2 * (
                    (((reduction_2d + input_1d * (
                        triton_helpers.div_floor_integer(
                            20 + ((-8) * kernel_size_0) + ((-2) * kernel_size_0 * kernel_size_2 * kernel_size_2) + 
                            4 * kernel_size_0 * kernel_size_1 + 8 * kernel_size_0 * kernel_size_2 + 
                            kernel_size_0 * kernel_size_1 * kernel_size_2 * kernel_size_2 + 
                            ((-4) * kernel_size_0 * kernel_size_1 * kernel_size_2), 
                            21
                        )
                    )) // ((-8) + ((-2) * kernel_size_2 * kernel_size_2) + 4 * kernel_size_1 + 8 * kernel_size_2 + 
                           kernel_size_1 * kernel_size_2 * kernel_size_2 + ((-4) * kernel_size_1 * kernel_size_2))) % kernel_size_0)
                )) + ((-4) * kernel_size_1 * kernel_size_2 * input_0d) + \
                16 * kernel_size_1 * kernel_size_2 * kernel_size_2 * (
                    (((reduction_2d + input_1d * (
                        triton_helpers.div_floor_integer(
                            20 + ((-8) * kernel_size_0) + ((-2) * kernel_size_0 * kernel_size_2 * kernel_size_2) + 
                            4 * kernel_size_0 * kernel_size_1 + 8 * kernel_size_0 * kernel_size_2 + 
                            kernel_size_0 * kernel_size_1 * kernel_size_2 * kernel_size_2 + 
                            ((-4) * kernel_size_0 * kernel_size_1 * kernel_size_2), 
                            21
                        )
                    )) // ((-8) + ((-2) * kernel_size_2 * kernel_size_2) + 4 * kernel_size_1 + 8 * kernel_size_2 + 
                           kernel_size_1 * kernel_size_2 * kernel_size_2 + ((-4) * kernel_size_1 * kernel_size_2))) % kernel_size_0)
                ) + (((reduction_2d + input_1d * (
                    triton_helpers.div_floor_integer(
                        20 + ((-8) * kernel_size_0) + ((-2) * kernel_size_0 * kernel_size_2 * kernel_size_2) + 
                        4 * kernel_size_0 * kernel_size_1 + 8 * kernel_size_0 * kernel_size_2 + 
                        kernel_size_0 * kernel_size_1 * kernel_size_2 * kernel_size_2 + 
                        ((-4) * kernel_size_0 * kernel_size_1 * kernel_size_2), 
                        21
                    )
                )) % ((-2) + kernel_size_2)))), reduction_mask & temp_mask & input_mask, 
                eviction_policy='evict_last', other=0.0
        )

        temp_broadcast = tl.broadcast_to(temp_mask, [XBLOCK, RBLOCK])
        temp_sum_update = temp_sum + temp_broadcast
        temp_sum = tl.where(reduction_mask & input_mask, temp_sum_update, temp_sum)

    temp_result = tl.sum(temp_sum, 1)[:, None]
    tl.store(output_ptr + (input_3d), temp_result, input_mask)