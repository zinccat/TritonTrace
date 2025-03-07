# From: 11_ConvTranspose2d_BatchNorm_Tanh_MaxPool_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_max_pool2d_with_indices_native_group_norm_4(
    input_ptr, output_ptr_max, output_ptr_indices, output_ptr_mean, output_ptr_var, 
    num_elements_x, num_elements_r, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    num_elements_r = 16384
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements_x
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_indices_flat = x_indices

    mean_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    m2_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    weight_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)

    for r_offset in range(0, num_elements_r, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < num_elements_r
        r1 = (r_indices % 32)
        r2 = r_indices // 32
        r3 = r_indices

        input_val_0 = tl.load(input_ptr + (2 * r1 + 128 * r2 + 65536 * x_indices_flat), rmask & x_mask, eviction_policy='evict_last', other=0.0)
        input_val_1 = tl.load(input_ptr + (1 + 2 * r1 + 128 * r2 + 65536 * x_indices_flat), rmask & x_mask, eviction_policy='evict_last', other=0.0)
        input_val_2 = tl.load(input_ptr + (64 + 2 * r1 + 128 * r2 + 65536 * x_indices_flat), rmask & x_mask, eviction_policy='evict_last', other=0.0)
        input_val_3 = tl.load(input_ptr + (65 + 2 * r1 + 128 * r2 + 65536 * x_indices_flat), rmask & x_mask, eviction_policy='evict_last', other=0.0)

        max_val_1 = triton_helpers.maximum(input_val_1, input_val_0)
        max_val_2 = triton_helpers.maximum(input_val_2, max_val_1)
        max_val_3 = triton_helpers.maximum(input_val_3, max_val_2)

        index_mask_1 = input_val_1 > input_val_0
        index_mask_2 = tl.full([1, 1], 1, tl.int8)
        index_mask_0 = tl.full([1, 1], 0, tl.int8)
        index_mask_combined = tl.where(index_mask_1, index_mask_2, index_mask_0)

        index_mask_3 = input_val_2 > max_val_1
        index_mask_4 = tl.full([1, 1], 2, tl.int8)
        index_mask_combined = tl.where(index_mask_3, index_mask_4, index_mask_combined)

        index_mask_5 = input_val_3 > max_val_2
        index_mask_6 = tl.full([1, 1], 3, tl.int8)
        final_index_mask = tl.where(index_mask_5, index_mask_6, index_mask_combined)

        broadcast_max_val = tl.broadcast_to(max_val_3, [XBLOCK, RBLOCK])

        mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
            broadcast_max_val, mean_accumulator, m2_accumulator, weight_accumulator, r_offset == 0
        )

        mean_accumulator = tl.where(rmask & x_mask, mean_next, mean_accumulator)
        m2_accumulator = tl.where(rmask & x_mask, m2_next, m2_accumulator)
        weight_accumulator = tl.where(rmask & x_mask, weight_next, weight_accumulator)

        tl.store(output_ptr_max + (r3 + 16384 * x_indices_flat), max_val_3, rmask & x_mask)
        tl.store(output_ptr_indices + (r3 + 16384 * x_indices_flat), final_index_mask, rmask & x_mask)

    mean_final, m2_final, weight_final = triton_helpers.welford(
        mean_accumulator, m2_accumulator, weight_accumulator, 1
    )

    mean_final = mean_final[:, None]
    m2_final = m2_final[:, None]

    tl.store(output_ptr_mean + (x_indices_flat), mean_final, x_mask)
    tl.store(output_ptr_var + (x_indices_flat), m2_final, x_mask)

    num_elements_r_float = 16384.0
    variance = m2_final / num_elements_r_float
    epsilon = 1e-05
    adjusted_variance = variance + epsilon
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(adjusted_variance)

    tl.store(output_ptr_var + (x_indices_flat), inv_sqrt_variance, x_mask)