# From: 7_Conv3d_ReLU_LeakyReLU_GELU_Sigmoid_BiasAdd

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_sum_1(input_ptr, output_ptr, kernel_size_z, kernel_size_y, kernel_size_x, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    input_num_elements = 336
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_index_1d = input_index // 16
    input_index_2d = (input_index % 16)
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_index_flat = input_index

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_index_flat = reduction_index

        temp_index_1 = reduction_index_flat + input_index_1d * (
            triton_helpers.div_floor_integer(
                20 + ((-8) * kernel_size_z) + ((-2) * kernel_size_z * kernel_size_x * kernel_size_x) + 
                4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + 
                kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + 
                ((-4) * kernel_size_z * kernel_size_y * kernel_size_x), 
                21
            )
        )

        temp_index_2 = ((-8) * kernel_size_z) + ((-2) * kernel_size_z * kernel_size_x * kernel_size_x) + 
                       4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + 
                       kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + 
                       ((-4) * kernel_size_z * kernel_size_y * kernel_size_x)

        index_condition = temp_index_1 < temp_index_2

        temp_load = tl.load(
            input_ptr + (
                ((-128) * (
                    (((reduction_index_flat + input_index_1d * (
                        triton_helpers.div_floor_integer(
                            20 + ((-8) * kernel_size_z) + ((-2) * kernel_size_z * kernel_size_x * kernel_size_x) + 
                            4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + 
                            kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + 
                            ((-4) * kernel_size_z * kernel_size_y * kernel_size_x), 
                            21
                        )
                    ) // ((-8) + ((-2) * kernel_size_x * kernel_size_x) + 4 * kernel_size_y + 8 * kernel_size_x + 
                          kernel_size_y * kernel_size_x * kernel_size_x + ((-4) * kernel_size_y * kernel_size_x))) % kernel_size_z)
                ) + ((-8) * input_index_2d) + ((-2) * (
                    (((reduction_index_flat + input_index_1d * (
                        triton_helpers.div_floor_integer(
                            20 + ((-8) * kernel_size_z) + ((-2) * kernel_size_z * kernel_size_x * kernel_size_x) + 
                            4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + 
                            kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + 
                            ((-4) * kernel_size_z * kernel_size_y * kernel_size_x), 
                            21
                        )
                    ) // ((-2) + kernel_size_x)) % ((-2) + kernel_size_x))
                )) + 4 * (
                    (((reduction_index_flat + input_index_1d * (
                        triton_helpers.div_floor_integer(
                            20 + ((-8) * kernel_size_z) + ((-2) * kernel_size_z * kernel_size_x * kernel_size_x) + 
                            4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + 
                            kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + 
                            ((-4) * kernel_size_z * kernel_size_y * kernel_size_x), 
                            21
                        )
                    ) // (4 + kernel_size_x * kernel_size_x + ((-4) * kernel_size_x))) % ((-2) + kernel_size_y))
                ) + kernel_size_x * (
                    (((reduction_index_flat + input_index_1d * (
                        triton_helpers.div_floor_integer(
                            20 + ((-8) * kernel_size_z) + ((-2) * kernel_size_z * kernel_size_x * kernel_size_x) + 
                            4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + 
                            kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + 
                            ((-4) * kernel_size_z * kernel_size_y * kernel_size_x), 
                            21
                        )
                    ) // ((-2) + kernel_size_x)) % ((-2) + kernel_size_x))
                ) + kernel_size_x * kernel_size_x * (
                    (((reduction_index_flat + input_index_1d * (
                        triton_helpers.div_floor_integer(
                            20 + ((-8) * kernel_size_z) + ((-2) * kernel_size_z * kernel_size_x * kernel_size_x) + 
                            4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + 
                            kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + 
                            ((-4) * kernel_size_z * kernel_size_y * kernel_size_x), 
                            21
                        )
                    ) // (4 + kernel_size_x * kernel_size_x + ((-4) * kernel_size_x))) % ((-2) + kernel_size_y))
                ) + ((-32) * kernel_size_x * kernel_size_x * (
                    (((reduction_index_flat + input_index_1d * (
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
                    (((reduction_index_flat + input_index_1d * (
                        triton_helpers.div_floor_integer(
                            20 + ((-8) * kernel_size_z) + ((-2) * kernel_size_z * kernel_size_x * kernel_size_x) + 
                            4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + 
                            kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + 
                            ((-4) * kernel_size_z * kernel_size_y * kernel_size_x), 
                            21
                        )
                    ) // (4 + kernel_size_x * kernel_size_x + ((-4) * kernel_size_x))) % ((-2) + kernel_size_y))
                )) + ((-2) * input_index_2d * kernel_size_x * kernel_size_x) + 
                4 * kernel_size_y * input_index_2d + 8 * kernel_size_x * input_index_2d + 
                64 * kernel_size_y * (
                    (((reduction_index_flat + input_index_1d * (
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
                    (((reduction_index_flat + input_index_1d * (
                        triton_helpers.div_floor_integer(
                            20 + ((-8) * kernel_size_z) + ((-2) * kernel_size_z * kernel_size_x * kernel_size_x) + 
                            4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + 
                            kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + 
                            ((-4) * kernel_size_z * kernel_size_y * kernel_size_x), 
                            21
                        )
                    ) // ((-8) + ((-2) * kernel_size_x * kernel_size_x) + 4 * kernel_size_y + 8 * kernel_size_x + 
                          kernel_size_y * kernel_size_x * kernel_size_x + ((-4) * kernel_size_y * kernel_size_x))) % kernel_size_z)
                ) + kernel_size_y * input_index_2d * kernel_size_x * kernel_size_x + 
                ((-64) * kernel_size_y * kernel_size_x * (
                    (((reduction_index_flat + input_index_1d * (
                        triton_helpers.div_floor_integer(
                            20 + ((-8) * kernel_size_z) + ((-2) * kernel_size_z * kernel_size_x * kernel_size_x) + 
                            4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + 
                            kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + 
                            ((-4) * kernel_size_z * kernel_size_y * kernel_size_x), 
                            21
                        )
                    ) // ((-8) + ((-2) * kernel_size_x * kernel_size_x) + 4 * kernel_size_y + 8 * kernel_size_x + 
                          kernel_size_y * kernel_size_x * kernel_size_x + ((-4) * kernel_size_y * kernel_size_x))) % kernel_size_z)
                )) + ((-4) * kernel_size_y * kernel_size_x * input_index_2d) + 
                16 * kernel_size_y * kernel_size_x * kernel_size_x * (
                    (((reduction_index_flat + input_index_1d * (
                        triton_helpers.div_floor_integer(
                            20 + ((-8) * kernel_size_z) + ((-2) * kernel_size_z * kernel_size_x * kernel_size_x) + 
                            4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + 
                            kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + 
                            ((-4) * kernel_size_z * kernel_size_y * kernel_size_x), 
                            21
                        )
                    ) // ((-8) + ((-2) * kernel_size_x * kernel_size_x) + 4 * kernel_size_y + 8 * kernel_size_x + 
                          kernel_size_y * kernel_size_x * kernel_size_x + ((-4) * kernel_size_y * kernel_size_x))) % kernel_size_z)
                )) + (((reduction_index_flat + input_index_1d * (
                    triton_helpers.div_floor_integer(
                        20 + ((-8) * kernel_size_z) + ((-2) * kernel_size_z * kernel_size_x * kernel_size_x) + 
                        4 * kernel_size_z * kernel_size_y + 8 * kernel_size_z * kernel_size_x + 
                        kernel_size_z * kernel_size_y * kernel_size_x * kernel_size_x + 
                        ((-4) * kernel_size_z * kernel_size_y * kernel_size_x), 
                        21
                    )
                )) % ((-2) + kernel_size_x)))), reduction_mask & index_condition & input_mask, eviction_policy='evict_last', other=0.0)

        temp_broadcast = tl.broadcast_to(index_condition, [XBLOCK, RBLOCK])
        temp_accumulator_update = temp_accumulator + temp_broadcast
        temp_accumulator = tl.where(reduction_mask & input_mask, temp_accumulator_update, temp_accumulator)

    temp_sum = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(output_ptr + (input_index_flat), temp_sum, input_mask)