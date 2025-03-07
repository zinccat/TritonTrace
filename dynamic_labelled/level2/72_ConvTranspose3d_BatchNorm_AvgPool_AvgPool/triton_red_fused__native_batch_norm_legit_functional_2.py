# From: 72_ConvTranspose3d_BatchNorm_AvgPool_AvgPool

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_2(
    input_ptr, output_mean_ptr, output_var_ptr, output_weight_ptr, kernel_size_0, kernel_size_1, 
    input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    input_num_elements = 3984
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_x0 = (input_index % 249)
    input_x1 = input_index // 249
    mean_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    variance_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    weight_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    input_x3 = input_index

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_r2 = reduction_index

        tmp0 = reduction_r2 + input_x0 * (
            triton_helpers.div_floor_integer(
                248 + ((-1) * kernel_size_0) + ((-12) * kernel_size_0 * kernel_size_1 * kernel_size_1) + 
                6 * kernel_size_0 * kernel_size_1 + 8 * kernel_size_0 * kernel_size_1 * kernel_size_1 * kernel_size_1, 
                249
            )
        )
        tmp1 = ((-1) * kernel_size_0) + ((-12) * kernel_size_0 * kernel_size_1 * kernel_size_1) + 
               6 * kernel_size_0 * kernel_size_1 + 8 * kernel_size_0 * kernel_size_1 * kernel_size_1 * kernel_size_1
        tmp2 = tmp0 < tmp1

        tmp3 = tl.load(
            input_ptr + (
                ((-1) * input_x1) + 
                ((-1) * (((reduction_r2 + input_x0 * (
                    triton_helpers.div_floor_integer(
                        248 + ((-1) * kernel_size_0) + ((-12) * kernel_size_0 * kernel_size_1 * kernel_size_1) + 
                        6 * kernel_size_0 * kernel_size_1 + 8 * kernel_size_0 * kernel_size_1 * kernel_size_1 * kernel_size_1, 
                        249
                    )
                )) // ((-1) + 2 * kernel_size_1)) % ((-1) + 2 * kernel_size_1))) + 
                ((-16) * (((reduction_r2 + input_x0 * (
                    triton_helpers.div_floor_integer(
                        248 + ((-1) * kernel_size_0) + ((-12) * kernel_size_0 * kernel_size_1 * kernel_size_1) + 
                        6 * kernel_size_0 * kernel_size_1 + 8 * kernel_size_0 * kernel_size_1 * kernel_size_1 * kernel_size_1, 
                        249
                    )
                )) // ((-1) + ((-12) * kernel_size_1 * kernel_size_1) + 6 * kernel_size_1 + 8 * kernel_size_1 * kernel_size_1 * kernel_size_1)) % kernel_size_0))) + 
                ((-192) * kernel_size_1 * kernel_size_1 * ((((
                    reduction_r2 + input_x0 * (
                        triton_helpers.div_floor_integer(
                            248 + ((-1) * kernel_size_0) + ((-12) * kernel_size_0 * kernel_size_1 * kernel_size_1) + 
                            6 * kernel_size_0 * kernel_size_1 + 8 * kernel_size_0 * kernel_size_1 * kernel_size_1 * kernel_size_1, 
                            249
                        )
                    )
                )) // ((-1) + ((-12) * kernel_size_1 * kernel_size_1) + 6 * kernel_size_1 + 8 * kernel_size_1 * kernel_size_1 * kernel_size_1)) % kernel_size_0))) + 
                ((-12) * input_x1 * kernel_size_1 * kernel_size_1) + 
                ((-4) * kernel_size_1 * ((((
                    reduction_r2 + input_x0 * (
                        triton_helpers.div_floor_integer(
                            248 + ((-1) * kernel_size_0) + ((-12) * kernel_size_0 * kernel_size_1 * kernel_size_1) + 
                            6 * kernel_size_0 * kernel_size_1 + 8 * kernel_size_0 * kernel_size_1 * kernel_size_1 * kernel_size_1, 
                            249
                        )
                    )
                )) // (1 + ((-4) * kernel_size_1) + 4 * kernel_size_1 * kernel_size_1)) % ((-1) + 2 * kernel_size_1))) + 
                2 * kernel_size_1 * ((((
                    reduction_r2 + input_x0 * (
                        triton_helpers.div_floor_integer(
                            248 + ((-1) * kernel_size_0) + ((-12) * kernel_size_0 * kernel_size_1 * kernel_size_1) + 
                            6 * kernel_size_0 * kernel_size_1 + 8 * kernel_size_0 * kernel_size_1 * kernel_size_1 * kernel_size_1, 
                            249
                        )
                    )
                )) // ((-1) + 2 * kernel_size_1)) % ((-1) + 2 * kernel_size_1))) + 
                4 * kernel_size_1 * kernel_size_1 * ((((
                    reduction_r2 + input_x0 * (
                        triton_helpers.div_floor_integer(
                            248 + ((-1) * kernel_size_0) + ((-12) * kernel_size_0 * kernel_size_1 * kernel_size_1) + 
                            6 * kernel_size_0 * kernel_size_1 + 8 * kernel_size_0 * kernel_size_1 * kernel_size_1 * kernel_size_1, 
                            249
                        )
                    )
                )) // (1 + ((-4) * kernel_size_1) + 4 * kernel_size_1 * kernel_size_1)) % ((-1) + 2 * kernel_size_1))) + 
                6 * kernel_size_1 * input_x1 + 
                8 * input_x1 * kernel_size_1 * kernel_size_1 * kernel_size_1 + 
                96 * kernel_size_1 * ((((
                    reduction_r2 + input_x0 * (
                        triton_helpers.div_floor_integer(
                            248 + ((-1) * kernel_size_0) + ((-12) * kernel_size_0 * kernel_size_1 * kernel_size_1) + 
                            6 * kernel_size_0 * kernel_size_1 + 8 * kernel_size_0 * kernel_size_1 * kernel_size_1 * kernel_size_1, 
                            249
                        )
                    )
                )) // ((-1) + ((-12) * kernel_size_1 * kernel_size_1) + 6 * kernel_size_1 + 8 * kernel_size_1 * kernel_size_1 * kernel_size_1)) % kernel_size_0)) + 
                128 * kernel_size_1 * kernel_size_1 * kernel_size_1 * ((((
                    reduction_r2 + input_x0 * (
                        triton_helpers.div_floor_integer(
                            248 + ((-1) * kernel_size_0) + ((-12) * kernel_size_0 * kernel_size_1 * kernel_size_1) + 
                            6 * kernel_size_0 * kernel_size_1 + 8 * kernel_size_0 * kernel_size_1 * kernel_size_1 * kernel_size_1, 
                            249
                        )
                    )
                )) // ((-1) + ((-12) * kernel_size_1 * kernel_size_1) + 6 * kernel_size_1 + 8 * kernel_size_1 * kernel_size_1 * kernel_size_1)) % kernel_size_0)) + 
                (((reduction_r2 + input_x0 * (
                    triton_helpers.div_floor_integer(
                        248 + ((-1) * kernel_size_0) + ((-12) * kernel_size_0 * kernel_size_1 * kernel_size_1) + 
                        6 * kernel_size_0 * kernel_size_1 + 8 * kernel_size_0 * kernel_size_1 * kernel_size_1 * kernel_size_1, 
                        249
                    )
                ))) % ((-1) + 2 * kernel_size_1)) + 
                ((((
                    reduction_r2 + input_x0 * (
                        triton_helpers.div_floor_integer(
                            248 + ((-1) * kernel_size_0) + ((-12) * kernel_size_0 * kernel_size_1 * kernel_size_1) + 
                            6 * kernel_size_0 * kernel_size_1 + 8 * kernel_size_0 * kernel_size_1 * kernel_size_1 * kernel_size_1, 
                            249
                        )
                    )
                )) // (1 + ((-4) * kernel_size_1) + 4 * kernel_size_1 * kernel_size_1)) % ((-1) + 2 * kernel_size_1))
            ), 
            reduction_mask & tmp2 & input_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )

        zero_value = 0.0
        zero_tensor = tl.full(zero_value.shape, 0, zero_value.dtype)
        zero_or_one_tensor = tl.where(tmp2, zero_value, zero_tensor)

        one_value = 1.0
        one_tensor = tl.full(one_value.shape, 0, one_value.dtype)
        one_or_zero_tensor = tl.where(tmp2, one_value, one_tensor)

        broadcast_mask = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        broadcast_zero_or_one = tl.broadcast_to(zero_or_one_tensor, [XBLOCK, RBLOCK])
        broadcast_one_or_zero = tl.broadcast_to(one_or_zero_tensor, [XBLOCK, RBLOCK])

        mean_accumulator_next, variance_accumulator_next, weight_accumulator_next = triton_helpers.welford_combine(
            mean_accumulator, variance_accumulator, weight_accumulator,
            broadcast_mask, broadcast_zero_or_one, broadcast_one_or_zero
        )

        mean_accumulator = tl.where(reduction_mask & input_mask, mean_accumulator_next, mean_accumulator)
        variance_accumulator = tl.where(reduction_mask & input_mask, variance_accumulator_next, variance_accumulator)
        weight_accumulator = tl.where(reduction_mask & input_mask, weight_accumulator_next, weight_accumulator)

    mean_result, variance_result, weight_result = triton_helpers.welford(
        mean_accumulator, variance_accumulator, weight_accumulator, 1
    )

    mean_result = mean_result[:, None]
    variance_result = variance_result[:, None]
    weight_result = weight_result[:, None]

    tl.store(output_mean_ptr + (input_x3), mean_result, input_mask)
    tl.store(output_var_ptr + (input_x3), variance_result, input_mask)
    tl.store(output_weight_ptr + (input_x3), weight_result, input_mask)