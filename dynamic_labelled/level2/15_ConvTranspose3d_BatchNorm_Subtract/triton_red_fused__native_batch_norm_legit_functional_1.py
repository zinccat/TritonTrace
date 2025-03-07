# From: 15_ConvTranspose3d_BatchNorm_Subtract

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_1(
    input_ptr, output_mean_ptr, output_variance_ptr, output_count_ptr, kernel_size, 
    input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    input_num_elements = 352
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_block_1 = input_index // 32
    input_block_0 = (input_index % 32)
    mean_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    variance_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    count_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    input_block_3 = input_index

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_block_2 = reduction_index

        temp_index_0 = (
            reduction_block_2 
            + 46 * input_block_1 
            + input_block_1 * triton_helpers.div_floor_integer(
                ((-1984) * kernel_size) + 1984 * kernel_size * kernel_size, 11
            )
        )
        temp_index_1 = 496 + ((-1984) * kernel_size) + 1984 * kernel_size * kernel_size
        temp_mask = temp_index_0 < temp_index_1

        input_data = tl.load(
            input_ptr + (
                -1 * (
                    (
                        (reduction_block_2 
                        + 46 * input_block_1 
                        + input_block_1 * triton_helpers.div_floor_integer(
                            ((-1984) * kernel_size) + 1984 * kernel_size * kernel_size, 11
                        )) // ((-1) + 2 * kernel_size)
                    ) % ((-1) + 2 * kernel_size)
                ) 
                + 31 * input_block_0 
                + 992 * (
                    (
                        (reduction_block_2 
                        + 46 * input_block_1 
                        + input_block_1 * triton_helpers.div_floor_integer(
                            ((-1984) * kernel_size) + 1984 * kernel_size * kernel_size, 11
                        )) // (31 + ((-124) * kernel_size) + 124 * kernel_size * kernel_size)
                    ) % 16
                ) 
                + ((-3968) * kernel_size * (
                    (
                        (reduction_block_2 
                        + 46 * input_block_1 
                        + input_block_1 * triton_helpers.div_floor_integer(
                            ((-1984) * kernel_size) + 1984 * kernel_size * kernel_size, 11
                        )) // (31 + ((-124) * kernel_size) + 124 * kernel_size * kernel_size)
                    ) % 16
                )) 
                + ((-124) * kernel_size * input_block_0) 
                + ((-4) * kernel_size * (
                    (
                        (reduction_block_2 
                        + 46 * input_block_1 
                        + input_block_1 * triton_helpers.div_floor_integer(
                            ((-1984) * kernel_size) + 1984 * kernel_size * kernel_size, 11
                        )) // (1 + ((-4) * kernel_size) + 4 * kernel_size * kernel_size)
                    ) % 31
                )) 
                + 2 * kernel_size * (
                    (
                        (reduction_block_2 
                        + 46 * input_block_1 
                        + input_block_1 * triton_helpers.div_floor_integer(
                            ((-1984) * kernel_size) + 1984 * kernel_size * kernel_size, 11
                        )) // ((-1) + 2 * kernel_size)
                    ) % ((-1) + 2 * kernel_size)
                ) 
                + 4 * kernel_size * kernel_size * (
                    (
                        (reduction_block_2 
                        + 46 * input_block_1 
                        + input_block_1 * triton_helpers.div_floor_integer(
                            ((-1984) * kernel_size) + 1984 * kernel_size * kernel_size, 11
                        )) // (1 + ((-4) * kernel_size) + 4 * kernel_size * kernel_size)
                    ) % 31
                ) 
                + 124 * input_block_0 * kernel_size * kernel_size 
                + 3968 * kernel_size * kernel_size * (
                    (
                        (reduction_block_2 
                        + 46 * input_block_1 
                        + input_block_1 * triton_helpers.div_floor_integer(
                            ((-1984) * kernel_size) + 1984 * kernel_size * kernel_size, 11
                        )) // (31 + ((-124) * kernel_size) + 124 * kernel_size * kernel_size)
                    ) % 16
                ) 
                + (
                    (reduction_block_2 
                    + 46 * input_block_1 
                    + input_block_1 * triton_helpers.div_floor_integer(
                        ((-1984) * kernel_size) + 1984 * kernel_size * kernel_size, 11
                    )) % ((-1) + 2 * kernel_size)
                ) 
                + (
                    (
                        (reduction_block_2 
                        + 46 * input_block_1 
                        + input_block_1 * triton_helpers.div_floor_integer(
                            ((-1984) * kernel_size) + 1984 * kernel_size * kernel_size, 11
                        )) // (1 + ((-4) * kernel_size) + 4 * kernel_size * kernel_size)
                    ) % 31
                )
            ), 
            reduction_mask & temp_mask & input_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )

        zero_value = 0.0
        zero_broadcast = tl.full(zero_value.shape, 0, zero_value.dtype)
        temp_value_6 = tl.where(temp_mask, zero_value, zero_broadcast)

        one_value = 1.0
        one_broadcast = tl.full(one_value.shape, 0, one_value.dtype)
        temp_value_9 = tl.where(temp_mask, one_value, one_broadcast)

        temp_mask_broadcast = tl.broadcast_to(temp_mask, [XBLOCK, RBLOCK])
        temp_value_6_broadcast = tl.broadcast_to(temp_value_6, [XBLOCK, RBLOCK])
        temp_value_9_broadcast = tl.broadcast_to(temp_value_9, [XBLOCK, RBLOCK])

        mean_accumulator_next, variance_accumulator_next, count_accumulator_next = triton_helpers.welford_combine(
            mean_accumulator, variance_accumulator, count_accumulator,
            temp_mask_broadcast, temp_value_6_broadcast, temp_value_9_broadcast
        )

        mean_accumulator = tl.where(reduction_mask & input_mask, mean_accumulator_next, mean_accumulator)
        variance_accumulator = tl.where(reduction_mask & input_mask, variance_accumulator_next, variance_accumulator)
        count_accumulator = tl.where(reduction_mask & input_mask, count_accumulator_next, count_accumulator)

    mean_result, variance_result, count_result = triton_helpers.welford(
        mean_accumulator, variance_accumulator, count_accumulator, 1
    )

    mean_result_broadcast = mean_result[:, None]
    variance_result_broadcast = variance_result[:, None]
    count_result_broadcast = count_result[:, None]

    tl.store(output_mean_ptr + (input_block_3), mean_result_broadcast, input_mask)
    tl.store(output_variance_ptr + (input_block_3), variance_result_broadcast, input_mask)
    tl.store(output_count_ptr + (input_block_3), count_result_broadcast, input_mask)