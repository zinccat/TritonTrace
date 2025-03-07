# From: 77_ConvTranspose3d_Scale_BatchNorm_GlobalAvgPool

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_mul_1(
    input_ptr, output_mean_ptr, output_var_ptr, output_weight_ptr, 
    num_elements, num_reduction_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    num_elements = 352
    num_reduction_elements = 30267
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_div_32 = x_indices // 32
    x_mod_32 = x_indices % 32
    temp_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x_indices_flat = x_indices

    for r_offset in range(0, num_reduction_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < num_reduction_elements
        r_indices_flat = r_indices
        temp_index = r_indices_flat + (num_reduction_elements * x_div_32)
        max_index = tl.full([1, 1], 332928, tl.int32)
        valid_mask = temp_index < max_index
        loaded_values = tl.load(
            input_ptr + ((20808 * x_mod_32) + (665856 * (((r_indices_flat + (num_reduction_elements * x_div_32)) // 20808) % 16)) + ((r_indices_flat + (num_reduction_elements * x_div_32)) % 20808)),
            valid_mask & x_mask,
            eviction_policy='evict_last',
            other=0.0
        )
        scaled_values = loaded_values * 2.0
        zero_filled_values = tl.full(scaled_values.shape, 0, scaled_values.dtype)
        valid_scaled_values = tl.where(valid_mask, scaled_values, zero_filled_values)
        zero_mask = tl.full(0.0.shape, 0, 0.0.dtype)
        valid_zero_mask = tl.where(valid_mask, 0.0, zero_mask)
        one_mask = tl.full(1.0.shape, 0, 1.0.dtype)
        valid_one_mask = tl.where(valid_mask, 1.0, one_mask)
        broadcasted_values = tl.broadcast_to(valid_scaled_values, [XBLOCK, RBLOCK])
        broadcasted_zeros = tl.broadcast_to(valid_zero_mask, [XBLOCK, RBLOCK])
        broadcasted_ones = tl.broadcast_to(valid_one_mask, [XBLOCK, RBLOCK])
        temp_mean_next, temp_m2_next, temp_weight_next = triton_helpers.welford_combine(
            temp_mean, temp_m2, temp_weight,
            broadcasted_values, broadcasted_zeros, broadcasted_ones
        )
        temp_mean = tl.where(r_mask & x_mask, temp_mean_next, temp_mean)
        temp_m2 = tl.where(r_mask & x_mask, temp_m2_next, temp_m2)
        temp_weight = tl.where(r_mask & x_mask, temp_weight_next, temp_weight)

    final_mean, final_m2, final_weight = triton_helpers.welford(
        temp_mean, temp_m2, temp_weight, 1
    )
    final_mean = final_mean[:, None]
    final_m2 = final_m2[:, None]
    final_weight = final_weight[:, None]
    tl.store(output_mean_ptr + (x_indices_flat), final_mean, x_mask)
    tl.store(output_var_ptr + (x_indices_flat), final_m2, x_mask)
    tl.store(output_weight_ptr + (x_indices_flat), final_weight, x_mask)