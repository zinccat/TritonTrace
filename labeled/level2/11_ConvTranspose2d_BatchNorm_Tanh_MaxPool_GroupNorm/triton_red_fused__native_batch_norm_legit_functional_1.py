# From: 11_ConvTranspose2d_BatchNorm_Tanh_MaxPool_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_1(
    input_ptr, output_mean_ptr, output_var_ptr, output_weight_ptr, 
    num_elements, num_reduction_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    num_elements = 384
    num_reduction_elements = 87382
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_channel = (x_indices // 64)
    x_within_channel = x_indices % 64
    temp_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x_flat_index = x_indices

    for r_offset in range(0, num_reduction_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < num_reduction_elements
        r_flat_index = r_indices
        temp_index = r_flat_index + (num_reduction_elements * x_channel)
        max_index = tl.full([1, 1], 524288, tl.int32)
        valid_index_mask = temp_index < max_index
        loaded_values = tl.load(
            input_ptr + ((4096 * x_within_channel) + 
                         (262144 * (((r_flat_index + (num_reduction_elements * x_channel)) // 4096) % 128)) + 
                         ((r_flat_index + (num_reduction_elements * x_channel)) % 4096)),
            valid_index_mask & x_mask, eviction_policy='evict_last', other=0.0
        )
        zero_values = tl.full(loaded_values.shape, 0, loaded_values.dtype)
        valid_values = tl.where(valid_index_mask, 0.0, zero_values)
        valid_mask = tl.where(valid_index_mask, 1.0, zero_values)
        broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
        broadcasted_zeros = tl.broadcast_to(valid_values, [XBLOCK, RBLOCK])
        broadcasted_ones = tl.broadcast_to(valid_mask, [XBLOCK, RBLOCK])
        
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
    
    tl.store(output_mean_ptr + (x_flat_index), final_mean, x_mask)
    tl.store(output_var_ptr + (x_flat_index), final_m2, x_mask)
    tl.store(output_weight_ptr + (x_flat_index), final_weight, x_mask)