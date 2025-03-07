# From: 11_ConvTranspose2d_BatchNorm_Tanh_MaxPool_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_max_pool2d_with_indices_native_group_norm_4(
    input_ptr, output_ptr_max, output_ptr_indices, output_ptr_mean, output_ptr_var, output_ptr_inv_std,
    num_elements_x, num_elements_r, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    num_elements_r = 16384
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements_x
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_indices_flat = x_indices

    running_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)

    for r_offset in range(0, num_elements_r, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < num_elements_r
        r_index_mod = r_indices % 32
        r_index_div = r_indices // 32
        r_indices_flat = r_indices

        input_val_0 = tl.load(input_ptr + (2 * r_index_mod + 128 * r_index_div + 65536 * x_indices_flat), r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        input_val_1 = tl.load(input_ptr + (1 + 2 * r_index_mod + 128 * r_index_div + 65536 * x_indices_flat), r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        input_val_2 = tl.load(input_ptr + (64 + 2 * r_index_mod + 128 * r_index_div + 65536 * x_indices_flat), r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        input_val_3 = tl.load(input_ptr + (65 + 2 * r_index_mod + 128 * r_index_div + 65536 * x_indices_flat), r_mask & x_mask, eviction_policy='evict_last', other=0.0)

        max_val_1_2 = triton_helpers.maximum(input_val_1, input_val_0)
        max_val_2_3 = triton_helpers.maximum(input_val_2, max_val_1_2)
        max_val_3_4 = triton_helpers.maximum(input_val_3, max_val_2_3)

        index_1_greater = input_val_1 > input_val_0
        index_1_mask = tl.full([1, 1], 1, tl.int8)
        index_0_mask = tl.full([1, 1], 0, tl.int8)
        index_mask_1 = tl.where(index_1_greater, index_1_mask, index_0_mask)

        index_2_greater = input_val_2 > max_val_1_2
        index_2_mask = tl.full([1, 1], 2, tl.int8)
        index_mask_2 = tl.where(index_2_greater, index_2_mask, index_mask_1)

        index_3_greater = input_val_3 > max_val_2_3
        index_3_mask = tl.full([1, 1], 3, tl.int8)
        index_mask_final = tl.where(index_3_greater, index_3_mask, index_mask_2)

        broadcast_max_val = tl.broadcast_to(max_val_3_4, [XBLOCK, RBLOCK])

        running_mean_next, running_m2_next, running_weight_next = triton_helpers.welford_reduce(
            broadcast_max_val, running_mean, running_m2, running_weight, r_offset == 0
        )

        running_mean = tl.where(r_mask & x_mask, running_mean_next, running_mean)
        running_m2 = tl.where(r_mask & x_mask, running_m2_next, running_m2)
        running_weight = tl.where(r_mask & x_mask, running_weight_next, running_weight)

        tl.store(output_ptr_max + (r_indices_flat + 16384 * x_indices_flat), max_val_3_4, r_mask & x_mask)
        tl.store(output_ptr_indices + (r_indices_flat + 16384 * x_indices_flat), index_mask_final, r_mask & x_mask)

    final_mean, final_m2, final_weight = triton_helpers.welford(
        running_mean, running_m2, running_weight, 1
    )

    final_mean_broadcast = final_mean[:, None]
    final_m2_broadcast = final_m2[:, None]

    tl.store(output_ptr_mean + (x_indices_flat), final_mean_broadcast, x_mask)
    tl.store(output_ptr_var + (x_indices_flat), final_m2_broadcast, x_mask)

    num_elements_r_float = 16384.0
    variance = final_m2_broadcast / num_elements_r_float
    epsilon = 1e-05
    variance_with_epsilon = variance + epsilon
    inv_std = tl.extra.cuda.libdevice.rsqrt(variance_with_epsilon)

    tl.store(output_ptr_inv_std + (x_indices_flat), inv_std, x_mask)