# From: 15_ConvTranspose3d_BatchNorm_Subtract

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_1(input_ptr, output_mean_ptr, output_variance_ptr, output_count_ptr, num_elements, num_reduced_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    num_elements = 352
    num_reduced_elements = 178966
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

    for r_offset in range(0, num_reduced_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < num_reduced_elements
        r_indices_flat = r_indices
        temp_index = r_indices_flat + (num_reduced_elements * x_div_32)
        max_index = tl.full([1, 1], 1968624, tl.int32)
        index_mask = temp_index < max_index
        loaded_values = tl.load(input_ptr + ((123039 * x_mod_32) + (3937248 * (((r_indices_flat + (num_reduced_elements * x_div_32)) // 123039) % 16)) + ((r_indices_flat + (num_reduced_elements * x_div_32)) % 123039)), r_mask & index_mask & x_mask, eviction_policy='evict_last', other=0.0)
        zero_values = tl.full(loaded_values.shape, 0, loaded_values.dtype)
        ones_values = tl.full(zero_values.shape, 1, zero_values.dtype)
        broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
        broadcasted_zeros = tl.broadcast_to(zero_values, [XBLOCK, RBLOCK])
        broadcasted_ones = tl.broadcast_to(ones_values, [XBLOCK, RBLOCK])
        
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
    
    final_mean_broadcast = final_mean[:, None]
    final_m2_broadcast = final_m2[:, None]
    final_weight_broadcast = final_weight[:, None]
    
    tl.store(output_mean_ptr + (x_indices_flat), final_mean_broadcast, x_mask)
    tl.store(output_variance_ptr + (x_indices_flat), final_m2_broadcast, x_mask)
    tl.store(output_count_ptr + (x_indices_flat), final_weight_broadcast, x_mask)