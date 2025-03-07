# From: 73_Conv2d_BatchNorm_Scaling

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_1red_fused__native_batch_norm_legit_functional_1(
    input_ptr, output_mean_ptr, output_var_ptr, output_weight_ptr, kernel_size_0, kernel_size_1, 
    input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    input_num_elements = 240
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_index_1 = input_index // 16
    input_index_0 = (input_index % 16)
    temp_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    input_index_3 = input_index

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_index_2 = reduction_index
        temp_index_0 = reduction_index_2 + input_index_1 * (
            triton_helpers.div_floor_integer(14 + 4 * kernel_size_0 + kernel_size_0 * kernel_size_1 * kernel_size_1 + (-4) * kernel_size_0 * kernel_size_1, 15)
        )
        temp_index_1 = 4 * kernel_size_0 + kernel_size_0 * kernel_size_1 * kernel_size_1 + (-4) * kernel_size_0 * kernel_size_1
        temp_mask = temp_index_0 < temp_index_1
        loaded_value = tl.load(
            input_ptr + (
                (-2) * (
                    (((reduction_index_2 + input_index_1 * (
                        triton_helpers.div_floor_integer(14 + 4 * kernel_size_0 + kernel_size_0 * kernel_size_1 * kernel_size_1 + (-4) * kernel_size_0 * kernel_size_1, 15)
                    )) // ((-2) + kernel_size_1)) % ((-2) + kernel_size_1))
                ) + 4 * input_index_0 + 64 * (
                    (((reduction_index_2 + input_index_1 * (
                        triton_helpers.div_floor_integer(14 + 4 * kernel_size_0 + kernel_size_0 * kernel_size_1 * kernel_size_1 + (-4) * kernel_size_0 * kernel_size_1, 15)
                    )) // (4 + kernel_size_1 * kernel_size_1 + (-4) * kernel_size_1)) % kernel_size_0)
                ) + kernel_size_1 * (
                    (((reduction_index_2 + input_index_1 * (
                        triton_helpers.div_floor_integer(14 + 4 * kernel_size_0 + kernel_size_0 * kernel_size_1 * kernel_size_1 + (-4) * kernel_size_0 * kernel_size_1, 15)
                    )) // ((-2) + kernel_size_1)) % ((-2) + kernel_size_1))
                ) + input_index_0 * kernel_size_1 * kernel_size_1 + (-64) * kernel_size_1 * (
                    (((reduction_index_2 + input_index_1 * (
                        triton_helpers.div_floor_integer(14 + 4 * kernel_size_0 + kernel_size_0 * kernel_size_1 * kernel_size_1 + (-4) * kernel_size_0 * kernel_size_1, 15)
                    )) // (4 + kernel_size_1 * kernel_size_1 + (-4) * kernel_size_1)) % kernel_size_0)
                ) + (-4) * kernel_size_1 * input_index_0 + 16 * kernel_size_1 * kernel_size_1 * (
                    (((reduction_index_2 + input_index_1 * (
                        triton_helpers.div_floor_integer(14 + 4 * kernel_size_0 + kernel_size_0 * kernel_size_1 * kernel_size_1 + (-4) * kernel_size_0 * kernel_size_1, 15)
                    )) // (4 + kernel_size_1 * kernel_size_1 + (-4) * kernel_size_1)) % kernel_size_0)
                ) + ((reduction_index_2 + input_index_1 * (
                    triton_helpers.div_floor_integer(14 + 4 * kernel_size_0 + kernel_size_0 * kernel_size_1 * kernel_size_1 + (-4) * kernel_size_0 * kernel_size_1, 15)
                )) % ((-2) + kernel_size_1))
            ), reduction_mask & temp_mask & input_mask, eviction_policy='evict_last', other=0.0
        )
        temp_zero = 0.0
        temp_zero_full = tl.full(temp_zero.shape, 0, temp_zero.dtype)
        temp_one = 1.0
        temp_one_full = tl.full(temp_one.shape, 0, temp_one.dtype)
        temp_broadcasted_value = tl.where(temp_mask, temp_zero, temp_zero_full)
        temp_broadcasted_one = tl.where(temp_mask, temp_one, temp_one_full)
        temp_broadcasted_loaded_value = tl.broadcast_to(loaded_value, [XBLOCK, RBLOCK])
        temp_broadcasted_temp_zero = tl.broadcast_to(temp_broadcasted_value, [XBLOCK, RBLOCK])
        temp_broadcasted_temp_one = tl.broadcast_to(temp_broadcasted_one, [XBLOCK, RBLOCK])
        temp_mean_next, temp_m2_next, temp_weight_next = triton_helpers.welford_combine(
            temp_mean, temp_m2, temp_weight,
            tl.broadcast_to(loaded_value, [XBLOCK, RBLOCK]),
            temp_broadcasted_temp_zero,
            temp_broadcasted_temp_one
        )
        temp_mean = tl.where(reduction_mask & input_mask, temp_mean_next, temp_mean)
        temp_m2 = tl.where(reduction_mask & input_mask, temp_m2_next, temp_m2)
        temp_weight = tl.where(reduction_mask & input_mask, temp_weight_next, temp_weight)

    temp_mean_final, temp_m2_final, temp_weight_final = triton_helpers.welford(
        temp_mean, temp_m2, temp_weight, 1
    )
    temp_mean_final_broadcast = temp_mean_final[:, None]
    temp_m2_final_broadcast = temp_m2_final[:, None]
    temp_weight_final_broadcast = temp_weight_final[:, None]
    tl.store(output_mean_ptr + (input_index_3), temp_mean_final_broadcast, input_mask)
    tl.store(output_var_ptr + (input_index_3), temp_m2_final_broadcast, input_mask)
    tl.store(output_weight_ptr + (input_index_3), temp_weight_final_broadcast, input_mask)