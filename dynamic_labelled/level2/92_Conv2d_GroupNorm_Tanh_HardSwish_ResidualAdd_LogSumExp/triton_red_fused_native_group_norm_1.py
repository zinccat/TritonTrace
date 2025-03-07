# From: 92_Conv2d_GroupNorm_Tanh_HardSwish_ResidualAdd_LogSumExp

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_group_norm_1red_fused_native_group_norm_1(
    in_out_ptr, input_ptr, output_ptr, kernel_size, num_elements_x, num_elements_r, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
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
        r_indices_flat = r_indices
        loaded_values = tl.load(
            input_ptr + (r_indices_flat + 8 * x_indices_flat + ((-8) * kernel_size * x_indices_flat) + 2 * x_indices_flat * kernel_size * kernel_size),
            r_mask & x_mask,
            eviction_policy='evict_first',
            other=0.0
        )
        broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
        mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
            broadcasted_values, mean_accumulator, m2_accumulator, weight_accumulator, r_offset == 0
        )
        mean_accumulator = tl.where(r_mask & x_mask, mean_next, mean_accumulator)
        m2_accumulator = tl.where(r_mask & x_mask, m2_next, m2_accumulator)
        weight_accumulator = tl.where(r_mask & x_mask, weight_next, weight_accumulator)

    mean_final, variance_final, weight_final = triton_helpers.welford(
        mean_accumulator, m2_accumulator, weight_accumulator, 1
    )
    mean_final_expanded = mean_final[:, None]
    variance_final_expanded = variance_final[:, None]
    weight_final_expanded = weight_final[:, None]

    tl.store(output_ptr + (x_indices_flat), mean_final_expanded, x_mask)

    offset_value = ((tl.full([], 0.0, tl.float64)) * ((tl.full([], 0.0, tl.float64)) >= (8 + ((-8) * kernel_size) + 2 * kernel_size * kernel_size)) + 
                    (8 + ((-8) * kernel_size) + 2 * kernel_size * kernel_size) * ((8 + ((-8) * kernel_size) + 2 * kernel_size * kernel_size) > (tl.full([], 0.0, tl.float64))))
    offset_value_float32 = offset_value.to(tl.float32)
    normalized_variance = variance_final_expanded / offset_value_float32
    epsilon = 1e-05
    adjusted_variance = normalized_variance + epsilon
    reciprocal_sqrt = tl.extra.cuda.libdevice.rsqrt(adjusted_variance)
    tl.debug_barrier()
    tl.store(in_out_ptr + (x_indices_flat), reciprocal_sqrt, x_mask)