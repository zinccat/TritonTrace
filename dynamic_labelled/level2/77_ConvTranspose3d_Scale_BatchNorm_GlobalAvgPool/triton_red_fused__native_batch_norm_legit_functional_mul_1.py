# From: 77_ConvTranspose3d_Scale_BatchNorm_GlobalAvgPool

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_mul_1red_fused__native_batch_norm_legit_functional_mul_1(
    input_ptr, output_mean_ptr, output_var_ptr, output_weight_ptr, kernel_size_0, kernel_size_1, total_elements, reduction_elements, 
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
    x_full_indices = x_indices

    for r_offset in range(0, reduction_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < reduction_elements
        r_index = r_indices
        temp_index = r_index + x_div_32 * ((10 + 4 * kernel_size_0 * kernel_size_0 + 8 * kernel_size_0 + 
                                            kernel_size_0 * kernel_size_0 * kernel_size_1 * kernel_size_1 + 
                                            2 * kernel_size_0 * kernel_size_1 * kernel_size_1 + 
                                            4 * kernel_size_1 * kernel_size_0 * kernel_size_0 + 
                                            8 * kernel_size_0 * kernel_size_1) // 11)
        temp_mask = (4 * kernel_size_0 * kernel_size_0 + 8 * kernel_size_0 + 
                     kernel_size_0 * kernel_size_0 * kernel_size_1 * kernel_size_1 + 
                     2 * kernel_size_0 * kernel_size_1 * kernel_size_1 + 
                     4 * kernel_size_1 * kernel_size_0 * kernel_size_0 + 
                     8 * kernel_size_0 * kernel_size_1)
        valid_mask = temp_index < temp_mask
        loaded_values = tl.load(input_ptr + (2 * (((temp_index // (2 + kernel_size_1)) % (2 + kernel_size_1)) + 
                                                 4 * ((temp_index // (4 + kernel_size_1 * kernel_size_1 + 4 * kernel_size_1)) % (2 + kernel_size_0)) + 
                                                 8 * x_mod_32 + 
                                                 256 * ((temp_index // (8 + 2 * kernel_size_1 * kernel_size_1 + 4 * kernel_size_0 + 
                                                                        8 * kernel_size_1 + kernel_size_0 * kernel_size_1 * kernel_size_1 + 
                                                                        4 * kernel_size_0 * kernel_size_1)) % kernel_size_0) + 
                                                 kernel_size_1 * (((temp_index // (2 + kernel_size_1)) % (2 + kernel_size_1))) + 
                                                 kernel_size_1 * kernel_size_1 * ((temp_index // (4 + kernel_size_1 * kernel_size_1 + 4 * kernel_size_1)) % (2 + kernel_size_0)) + 
                                                 2 * x_mod_32 * kernel_size_1 * kernel_size_1 + 
                                                 4 * kernel_size_0 * x_mod_32 + 
                                                 4 * kernel_size_1 * ((temp_index // (4 + kernel_size_1 * kernel_size_1 + 4 * kernel_size_1)) % (2 + kernel_size_0)) + 
                                                 8 * kernel_size_1 * x_mod_32 + 
                                                 64 * kernel_size_1 * kernel_size_1 * ((temp_index // (8 + 2 * kernel_size_1 * kernel_size_1 + 4 * kernel_size_0 + 
                                                                                                    8 * kernel_size_1 + kernel_size_0 * kernel_size_1 * kernel_size_1 + 
                                                                                                    4 * kernel_size_0 * kernel_size_1)) % kernel_size_0) + 
                                                 128 * kernel_size_0 * ((temp_index // (8 + 2 * kernel_size_1 * kernel_size_1 + 4 * kernel_size_0 + 
                                                                                               8 * kernel_size_1 + kernel_size_0 * kernel_size_1 * kernel_size_1 + 
                                                                                               4 * kernel_size_0 * kernel_size_1)) % kernel_size_0) + 
                                                 256 * kernel_size_1 * ((temp_index // (8 + 2 * kernel_size_1 * kernel_size_1 + 4 * kernel_size_0 + 
                                                                                               8 * kernel_size_1 + kernel_size_0 * kernel_size_1 * kernel_size_1 + 
                                                                                               4 * kernel_size_0 * kernel_size_1)) % kernel_size_0) + 
                                                 kernel_size_0 * x_mod_32 * kernel_size_1 * kernel_size_1 + 
                                                 4 * kernel_size_0 * kernel_size_1 * x_mod_32 + 
                                                 32 * kernel_size_0 * kernel_size_1 * kernel_size_1 * ((temp_index // (8 + 2 * kernel_size_1 * kernel_size_1 + 4 * kernel_size_0 + 
                                                                                                                                               8 * kernel_size_1 + kernel_size_0 * kernel_size_1 * kernel_size_1 + 
                                                                                                                                               4 * kernel_size_0 * kernel_size_1)) % kernel_size_0) + 
                                                 128 * kernel_size_0 * kernel_size_1 * ((temp_index // (8 + 2 * kernel_size_1 * kernel_size_1 + 4 * kernel_size_0 + 
                                                                                                           8 * kernel_size_1 + kernel_size_0 * kernel_size_1 * kernel_size_1 + 
                                                                                                           4 * kernel_size_0 * kernel_size_1)) % kernel_size_0) + 
                                                 (temp_index % (2 + kernel_size_1)))), r_mask & valid_mask & x_mask, eviction_policy='evict_last', other=0.0)
        
        scale_factor = 2.0
        scaled_values = loaded_values * scale_factor
        zero_tensor = tl.full(scaled_values.shape, 0, scaled_values.dtype)
        masked_values = tl.where(valid_mask, scaled_values, zero_tensor)
        zero_value = 0.0
        zero_broadcast = tl.full(zero_value.shape, 0, zero_value.dtype)
        masked_zeros = tl.where(valid_mask, zero_value, zero_broadcast)
        one_value = 1.0
        one_broadcast = tl.full(one_value.shape, 0, one_value.dtype)
        masked_ones = tl.where(valid_mask, one_value, one_broadcast)
        
        broadcasted_values = tl.broadcast_to(masked_values, [XBLOCK, RBLOCK])
        broadcasted_zeros = tl.broadcast_to(masked_zeros, [XBLOCK, RBLOCK])
        broadcasted_ones = tl.broadcast_to(masked_ones, [XBLOCK, RBLOCK])
        
        temp_mean_next, temp_m2_next, temp_weight_next = triton_helpers.welford_combine(
            temp_mean, temp_m2, temp_weight,
            broadcasted_values, broadcasted_zeros, broadcasted_ones
        )
        
        temp_mean = tl.where(r_mask & x_mask, temp_mean_next, temp_mean)
        temp_m2 = tl.where(r_mask & x_mask, temp_m2_next, temp_m2)
        temp_weight = tl.where(r_mask & x_mask, temp_weight_next, temp_weight)

    mean_result, variance_result, weight_result = triton_helpers.welford(
        temp_mean, temp_m2, temp_weight, 1
    )
    
    mean_result = mean_result[:, None]
    variance_result = variance_result[:, None]
    weight_result = weight_result[:, None]
    
    tl.store(output_mean_ptr + (x_full_indices), mean_result, x_mask)
    tl.store(output_var_ptr + (x_full_indices), variance_result, x_mask)
    tl.store(output_weight_ptr + (x_full_indices), weight_result, x_mask)