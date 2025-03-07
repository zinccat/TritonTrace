# From: 61_ConvTranspose3d_ReLU_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_group_norm_0(
    input_ptr, output_mean_ptr, output_var_ptr, output_weight_ptr, kernel_size_0, kernel_size_1, 
    num_elements_x, num_elements_r, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements_x
    r_base_indices = tl.arange(0, RBLOCK)[None, :]
    x_mod_4 = x_indices % 4
    x_div_4 = x_indices // 4
    temp_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x_indices_3d = x_indices

    for r_offset in range(0, num_elements_r, RBLOCK):
        r_indices = r_offset + r_base_indices
        r_mask = r_indices < num_elements_r
        r_indices_2d = r_indices

        temp_input = tl.load(
            input_ptr + (
                2 * (
                    (((r_indices_2d % (8 + 2 * kernel_size_0 * kernel_size_0 + 4 * kernel_size_1 + 8 * kernel_size_0 + kernel_size_1 * kernel_size_0 * kernel_size_0 + 4 * kernel_size_0 * kernel_size_1)) // (2 + kernel_size_0)) % (2 + kernel_size_0))
                ) + 4 * (
                    (((r_indices_2d % (8 + 2 * kernel_size_0 * kernel_size_0 + 4 * kernel_size_1 + 8 * kernel_size_0 + kernel_size_1 * kernel_size_0 * kernel_size_0 + 4 * kernel_size_0 * kernel_size_1)) // (4 + kernel_size_0 * kernel_size_0 + 4 * kernel_size_0)) % (2 + kernel_size_1))
                ) + 8 * (
                    (((r_indices_2d + 32 * x_mod_4 + 8 * x_mod_4 * kernel_size_0 * kernel_size_0 + 16 * kernel_size_1 * x_mod_4 + 32 * kernel_size_0 * x_mod_4 + 4 * kernel_size_1 * x_mod_4 * kernel_size_0 * kernel_size_0 + 16 * kernel_size_0 * kernel_size_1 * x_mod_4) // (8 + 2 * kernel_size_0 * kernel_size_0 + 4 * kernel_size_1 + 8 * kernel_size_0 + kernel_size_1 * kernel_size_0 * kernel_size_0 + 4 * kernel_size_0 * kernel_size_1)) % 16)
                ) + 128 * x_div_4 + kernel_size_0 * (
                    (((r_indices_2d % (8 + 2 * kernel_size_0 * kernel_size_0 + 4 * kernel_size_1 + 8 * kernel_size_0 + kernel_size_1 * kernel_size_0 * kernel_size_0 + 4 * kernel_size_0 * kernel_size_1)) // (2 + kernel_size_0)) % (2 + kernel_size_0))
                ) + kernel_size_0 * kernel_size_0 * (
                    (((r_indices_2d % (8 + 2 * kernel_size_0 * kernel_size_0 + 4 * kernel_size_1 + 8 * kernel_size_0 + kernel_size_1 * kernel_size_0 * kernel_size_0 + 4 * kernel_size_0 * kernel_size_1)) // (4 + kernel_size_0 * kernel_size_0 + 4 * kernel_size_0)) % (2 + kernel_size_1))
                ) + 2 * kernel_size_0 * kernel_size_0 * (
                    (((r_indices_2d + 32 * x_mod_4 + 8 * x_mod_4 * kernel_size_0 * kernel_size_0 + 16 * kernel_size_1 * x_mod_4 + 32 * kernel_size_0 * x_mod_4 + 4 * kernel_size_1 * x_mod_4 * kernel_size_0 * kernel_size_0 + 16 * kernel_size_0 * kernel_size_1 * x_mod_4) // (8 + 2 * kernel_size_0 * kernel_size_0 + 4 * kernel_size_1 + 8 * kernel_size_0 + kernel_size_1 * kernel_size_0 * kernel_size_0 + 4 * kernel_size_0 * kernel_size_1)) % 16)
                ) + 4 * kernel_size_0 * (
                    (((r_indices_2d % (8 + 2 * kernel_size_0 * kernel_size_0 + 4 * kernel_size_1 + 8 * kernel_size_0 + kernel_size_1 * kernel_size_0 * kernel_size_0 + 4 * kernel_size_0 * kernel_size_1)) // (4 + kernel_size_0 * kernel_size_0 + 4 * kernel_size_0)) % (2 + kernel_size_1))
                ) + 4 * kernel_size_1 * (
                    (((r_indices_2d + 32 * x_mod_4 + 8 * x_mod_4 * kernel_size_0 * kernel_size_0 + 16 * kernel_size_1 * x_mod_4 + 32 * kernel_size_0 * x_mod_4 + 4 * kernel_size_1 * x_mod_4 * kernel_size_0 * kernel_size_0 + 16 * kernel_size_0 * kernel_size_1 * x_mod_4) // (8 + 2 * kernel_size_0 * kernel_size_0 + 4 * kernel_size_1 + 8 * kernel_size_0 + kernel_size_1 * kernel_size_0 * kernel_size_0 + 4 * kernel_size_0 * kernel_size_1)) % 16)
                ) + 8 * kernel_size_0 * (
                    (((r_indices_2d + 32 * x_mod_4 + 8 * x_mod_4 * kernel_size_0 * kernel_size_0 + 16 * kernel_size_1 * x_mod_4 + 32 * kernel_size_0 * x_mod_4 + 4 * kernel_size_1 * x_mod_4 * kernel_size_0 * kernel_size_0 + 16 * kernel_size_0 * kernel_size_1 * x_mod_4) // (8 + 2 * kernel_size_0 * kernel_size_0 + 4 * kernel_size_1 + 8 * kernel_size_0 + kernel_size_1 * kernel_size_0 * kernel_size_0 + 4 * kernel_size_0 * kernel_size_1)) % 16)
                ) + 32 * x_div_4 * kernel_size_0 * kernel_size_0 + 64 * kernel_size_1 * x_div_4 + 128 * kernel_size_0 * x_div_4 + kernel_size_1 * kernel_size_0 * kernel_size_0 * (
                    (((r_indices_2d + 32 * x_mod_4 + 8 * x_mod_4 * kernel_size_0 * kernel_size_0 + 16 * kernel_size_1 * x_mod_4 + 32 * kernel_size_0 * x_mod_4 + 4 * kernel_size_1 * x_mod_4 * kernel_size_0 * kernel_size_0 + 16 * kernel_size_0 * kernel_size_1 * x_mod_4) // (8 + 2 * kernel_size_0 * kernel_size_0 + 4 * kernel_size_1 + 8 * kernel_size_0 + kernel_size_1 * kernel_size_0 * kernel_size_0 + 4 * kernel_size_0 * kernel_size_1)) % 16)
                ) + 4 * kernel_size_0 * kernel_size_1 * (
                    (((r_indices_2d + 32 * x_mod_4 + 8 * x_mod_4 * kernel_size_0 * kernel_size_0 + 16 * kernel_size_1 * x_mod_4 + 32 * kernel_size_0 * x_mod_4 + 4 * kernel_size_1 * x_mod_4 * kernel_size_0 * kernel_size_0 + 16 * kernel_size_0 * kernel_size_1 * x_mod_4) // (8 + 2 * kernel_size_0 * kernel_size_0 + 4 * kernel_size_1 + 8 * kernel_size_0 + kernel_size_1 * kernel_size_0 * kernel_size_0 + 4 * kernel_size_0 * kernel_size_1)) % 16)
                ) + 16 * kernel_size_1 * x_div_4 * kernel_size_0 * kernel_size_0 + 64 * kernel_size_0 * kernel_size_1 * x_div_4 + (((r_indices_2d % (8 + 2 * kernel_size_0 * kernel_size_0 + 4 * kernel_size_1 + 8 * kernel_size_0 + kernel_size_1 * kernel_size_0 * kernel_size_0 + 4 * kernel_size_0 * kernel_size_1)) % (2 + kernel_size_0)))), r_mask & x_mask, eviction_policy='evict_last', other=0.0
        )

        temp_max = tl.full([1, 1], 0, tl.int32)
        temp_clamped = triton_helpers.maximum(temp_max, temp_input)
        temp_broadcast = tl.broadcast_to(temp_clamped, [XBLOCK, RBLOCK])

        temp_mean_next, temp_m2_next, temp_weight_next = triton_helpers.welford_reduce(
            temp_broadcast, temp_mean, temp_m2, temp_weight, r_offset == 0
        )

        temp_mean = tl.where(r_mask & x_mask, temp_mean_next, temp_mean)
        temp_m2 = tl.where(r_mask & x_mask, temp_m2_next, temp_m2)
        temp_weight = tl.where(r_mask & x_mask, temp_weight_next, temp_weight)

    temp_mean_final, temp_var_final, temp_weight_final = triton_helpers.welford(
        temp_mean, temp_m2, temp_weight, 1
    )

    temp_mean_final = temp_mean_final[:, None]
    temp_var_final = temp_var_final[:, None]
    temp_weight_final = temp_weight_final[:, None]

    tl.store(output_mean_ptr + (x_indices_3d), temp_mean_final, x_mask)
    tl.store(output_var_ptr + (x_indices_3d), temp_var_final, x_mask)
    tl.store(output_weight_ptr + (x_indices_3d), temp_weight_final, x_mask)