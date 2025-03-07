# From: 73_Conv2d_BatchNorm_Scaling

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_1(
    input_ptr, output_mean_ptr, output_var_ptr, output_weight_ptr, kernel_size_0, kernel_size_1, 
    input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    input_num_elements = 240
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_block_row = input_index // 16
    input_block_col = (input_index % 16)
    mean_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    variance_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    weight_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    input_linear_index = input_index

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_linear_index = reduction_index

        combined_index = reduction_linear_index + input_block_row * (
            triton_helpers.div_floor_integer(14 + 4 * kernel_size_0 + kernel_size_0 * kernel_size_1 * kernel_size_1 + (-4) * kernel_size_0 * kernel_size_1, 15)
        )

        max_index = 4 * kernel_size_0 + kernel_size_0 * kernel_size_1 * kernel_size_1 + (-4) * kernel_size_0 * kernel_size_1
        valid_index_mask = combined_index < max_index

        input_data = tl.load(
            input_ptr + (
                -2 * (
                    (
                        (reduction_linear_index + input_block_row * (
                            triton_helpers.div_floor_integer(14 + 4 * kernel_size_0 + kernel_size_0 * kernel_size_1 * kernel_size_1 + (-4) * kernel_size_0 * kernel_size_1, 15)
                        )) // (-2 + kernel_size_1)
                    ) % (-2 + kernel_size_1)
                ) + 4 * input_block_col + 64 * (
                    (
                        (reduction_linear_index + input_block_row * (
                            triton_helpers.div_floor_integer(14 + 4 * kernel_size_0 + kernel_size_0 * kernel_size_1 * kernel_size_1 + (-4) * kernel_size_0 * kernel_size_1, 15)
                        )) // (4 + kernel_size_1 * kernel_size_1 + (-4) * kernel_size_1)
                    ) % kernel_size_0
                ) + kernel_size_1 * (
                    (
                        (reduction_linear_index + input_block_row * (
                            triton_helpers.div_floor_integer(14 + 4 * kernel_size_0 + kernel_size_0 * kernel_size_1 * kernel_size_1 + (-4) * kernel_size_0 * kernel_size_1, 15)
                        )) // (-2 + kernel_size_1)
                    ) % (-2 + kernel_size_1)
                ) + input_block_col * kernel_size_1 * kernel_size_1 + (-64) * kernel_size_1 * (
                    (
                        (reduction_linear_index + input_block_row * (
                            triton_helpers.div_floor_integer(14 + 4 * kernel_size_0 + kernel_size_0 * kernel_size_1 * kernel_size_1 + (-4) * kernel_size_0 * kernel_size_1, 15)
                        )) // (4 + kernel_size_1 * kernel_size_1 + (-4) * kernel_size_1)
                    ) % kernel_size_0
                ) + (-4) * kernel_size_1 * input_block_col + 16 * kernel_size_1 * kernel_size_1 * (
                    (
                        (reduction_linear_index + input_block_row * (
                            triton_helpers.div_floor_integer(14 + 4 * kernel_size_0 + kernel_size_0 * kernel_size_1 * kernel_size_1 + (-4) * kernel_size_0 * kernel_size_1, 15)
                        )) // (4 + kernel_size_1 * kernel_size_1 + (-4) * kernel_size_1)
                    ) % kernel_size_0
                ) + (
                    (reduction_linear_index + input_block_row * (
                        triton_helpers.div_floor_integer(14 + 4 * kernel_size_0 + kernel_size_0 * kernel_size_1 * kernel_size_1 + (-4) * kernel_size_0 * kernel_size_1, 15)
                    )) % (-2 + kernel_size_1)
                )
            ),
            reduction_mask & valid_index_mask & input_mask,
            eviction_policy='evict_last',
            other=0.0
        )

        zero_value = 0.0
        zero_broadcast = tl.full(zero_value.shape, 0, zero_value.dtype)
        zero_condition = tl.where(valid_index_mask, zero_value, zero_broadcast)

        one_value = 1.0
        one_broadcast = tl.full(one_value.shape, 0, one_value.dtype)
        one_condition = tl.where(valid_index_mask, one_value, one_broadcast)

        input_data_broadcast = tl.broadcast_to(input_data, [XBLOCK, RBLOCK])
        zero_condition_broadcast = tl.broadcast_to(zero_condition, [XBLOCK, RBLOCK])
        one_condition_broadcast = tl.broadcast_to(one_condition, [XBLOCK, RBLOCK])

        mean_accumulator_next, variance_accumulator_next, weight_accumulator_next = triton_helpers.welford_combine(
            mean_accumulator, variance_accumulator, weight_accumulator,
            input_data_broadcast, zero_condition_broadcast, one_condition_broadcast
        )

        mean_accumulator = tl.where(reduction_mask & input_mask, mean_accumulator_next, mean_accumulator)
        variance_accumulator = tl.where(reduction_mask & input_mask, variance_accumulator_next, variance_accumulator)
        weight_accumulator = tl.where(reduction_mask & input_mask, weight_accumulator_next, weight_accumulator)

    mean_result, variance_result, weight_result = triton_helpers.welford(
        mean_accumulator, variance_accumulator, weight_accumulator, 1
    )

    mean_result_broadcast = mean_result[:, None]
    variance_result_broadcast = variance_result[:, None]
    weight_result_broadcast = weight_result[:, None]

    tl.store(output_mean_ptr + (input_linear_index), mean_result_broadcast, input_mask)
    tl.store(output_var_ptr + (input_linear_index), variance_result_broadcast, input_mask)
    tl.store(output_weight_ptr + (input_linear_index), weight_result_broadcast, input_mask)