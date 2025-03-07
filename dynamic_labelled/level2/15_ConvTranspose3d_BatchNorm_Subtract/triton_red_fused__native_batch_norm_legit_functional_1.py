# From: 15_ConvTranspose3d_BatchNorm_Subtract

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_1red_fused__native_batch_norm_legit_functional_1(
    input_ptr, output_mean_ptr, output_variance_ptr, output_count_ptr, kernel_size, total_elements, reduction_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    total_elements = 352
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < total_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_div_32 = x_indices // 32
    x_mod_32 = x_indices % 32
    temp_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x_indices_flat = x_indices

    for r_offset in range(0, reduction_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < reduction_elements
        r_indices_flat = r_indices
        temp_index = r_indices_flat + 46 * x_div_32 + x_div_32 * (
            triton_helpers.div_floor_integer((-1984) * kernel_size + 1984 * kernel_size * kernel_size, 11)
        )
        temp_limit = 496 + ((-1984) * kernel_size) + 1984 * kernel_size * kernel_size
        temp_condition = temp_index < temp_limit
        temp_load = tl.load(
            input_ptr + (
                -1 * (
                    (
                        (r_indices_flat + 46 * x_div_32 + x_div_32 * (
                            triton_helpers.div_floor_integer(
                                (-1984) * kernel_size + 1984 * kernel_size * kernel_size, 11
                            )
                        )) // (-1 + 2 * kernel_size)
                    ) % (-1 + 2 * kernel_size)
                ) + 31 * x_mod_32 + 992 * (
                    (
                        (r_indices_flat + 46 * x_div_32 + x_div_32 * (
                            triton_helpers.div_floor_integer(
                                (-1984) * kernel_size + 1984 * kernel_size * kernel_size, 11
                            )
                        )) // (31 + (-124) * kernel_size + 124 * kernel_size * kernel_size)
                    ) % 16
                ) + (-3968) * kernel_size * (
                    (
                        (r_indices_flat + 46 * x_div_32 + x_div_32 * (
                            triton_helpers.div_floor_integer(
                                (-1984) * kernel_size + 1984 * kernel_size * kernel_size, 11
                            )
                        )) // (31 + (-124) * kernel_size + 124 * kernel_size * kernel_size)
                    ) % 16
                ) + (-124) * kernel_size * x_mod_32 + (-4) * kernel_size * (
                    (
                        (r_indices_flat + 46 * x_div_32 + x_div_32 * (
                            triton_helpers.div_floor_integer(
                                (-1984) * kernel_size + 1984 * kernel_size * kernel_size, 11
                            )
                        )) // (1 + (-4) * kernel_size + 4 * kernel_size * kernel_size)
                    ) % 31
                ) + 2 * kernel_size * (
                    (
                        (r_indices_flat + 46 * x_div_32 + x_div_32 * (
                            triton_helpers.div_floor_integer(
                                (-1984) * kernel_size + 1984 * kernel_size * kernel_size, 11
                            )
                        )) // (-1 + 2 * kernel_size)
                    ) % (-1 + 2 * kernel_size)
                ) + 4 * kernel_size * kernel_size * (
                    (
                        (r_indices_flat + 46 * x_div_32 + x_div_32 * (
                            triton_helpers.div_floor_integer(
                                (-1984) * kernel_size + 1984 * kernel_size * kernel_size, 11
                            )
                        )) // (1 + (-4) * kernel_size + 4 * kernel_size * kernel_size)
                    ) % 31
                ) + 124 * x_mod_32 * kernel_size * kernel_size + 3968 * kernel_size * kernel_size * (
                    (
                        (r_indices_flat + 46 * x_div_32 + x_div_32 * (
                            triton_helpers.div_floor_integer(
                                (-1984) * kernel_size + 1984 * kernel_size * kernel_size, 11
                            )
                        )) // (31 + (-124) * kernel_size + 124 * kernel_size * kernel_size)
                    ) % 16
                ) + (
                    (r_indices_flat + 46 * x_div_32 + x_div_32 * (
                        triton_helpers.div_floor_integer(
                            (-1984) * kernel_size + 1984 * kernel_size * kernel_size, 11
                        )
                    )) % (-1 + 2 * kernel_size)
                ) + (
                    (
                        (r_indices_flat + 46 * x_div_32 + x_div_32 * (
                            triton_helpers.div_floor_integer(
                                (-1984) * kernel_size + 1984 * kernel_size * kernel_size, 11
                            )
                        )) // (1 + (-4) * kernel_size + 4 * kernel_size * kernel_size)
                    ) % 31
                ),
            r_mask & temp_condition & x_mask,
            eviction_policy='evict_last',
            other=0.0
        )
        temp_zero = 0.0
        temp_full_zero = tl.full(temp_zero.shape, 0, temp_zero.dtype)
        temp_zero_broadcast = tl.where(temp_condition, temp_zero, temp_full_zero)
        temp_one = 1.0
        temp_full_one = tl.full(temp_one.shape, 0, temp_one.dtype)
        temp_one_broadcast = tl.where(temp_condition, temp_one, temp_full_one)
        temp_condition_broadcast = tl.broadcast_to(temp_condition, [XBLOCK, RBLOCK])
        temp_zero_broadcast_expanded = tl.broadcast_to(temp_zero_broadcast, [XBLOCK, RBLOCK])
        temp_one_broadcast_expanded = tl.broadcast_to(temp_one_broadcast, [XBLOCK, RBLOCK])
        temp_mean_next, temp_m2_next, temp_weight_next = triton_helpers.welford_combine(
            temp_mean, temp_m2, temp_weight,
            temp_condition_broadcast, temp_zero_broadcast_expanded, temp_one_broadcast_expanded
        )
        temp_mean = tl.where(r_mask & x_mask, temp_mean_next, temp_mean)
        temp_m2 = tl.where(r_mask & x_mask, temp_m2_next, temp_m2)
        temp_weight = tl.where(r_mask & x_mask, temp_weight_next, temp_weight)

    temp_mean_final, temp_m2_final, temp_weight_final = triton_helpers.welford(
        temp_mean, temp_m2, temp_weight, 1
    )
    temp_mean_final_expanded = temp_mean_final[:, None]
    temp_m2_final_expanded = temp_m2_final[:, None]
    temp_weight_final_expanded = temp_weight_final[:, None]
    tl.store(output_mean_ptr + (x_indices_flat), temp_mean_final_expanded, x_mask)
    tl.store(output_variance_ptr + (x_indices_flat), temp_m2_final_expanded, x_mask)
    tl.store(output_count_ptr + (x_indices_flat), temp_weight_final_expanded, x_mask)