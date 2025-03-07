# From: 77_ConvTranspose3d_Scale_BatchNorm_GlobalAvgPool

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_mul_1(
    input_ptr, output_mean_ptr, output_var_ptr, output_weight_ptr, kernel_size_0, kernel_size_1, 
    x_num_elements, r_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    x_num_elements = 352
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < x_num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_div_32 = x_index // 32
    x_mod_32 = x_index % 32
    tmp_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = x_index

    for r_offset in range(0, r_num_elements, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < r_num_elements
        r2 = r_index
        combined_index = r2 + x_div_32 * ((10 + 4 * kernel_size_0 * kernel_size_0 + 8 * kernel_size_0 + 
                                           kernel_size_0 * kernel_size_0 * kernel_size_1 * kernel_size_1 + 
                                           2 * kernel_size_0 * kernel_size_1 * kernel_size_1 + 
                                           4 * kernel_size_1 * kernel_size_0 * kernel_size_0 + 
                                           8 * kernel_size_0 * kernel_size_1) // 11)
        divisor = 4 * kernel_size_0 * kernel_size_0 + 8 * kernel_size_0 + kernel_size_0 * kernel_size_0 * kernel_size_1 * kernel_size_1 + \
                  2 * kernel_size_0 * kernel_size_1 * kernel_size_1 + 4 * kernel_size_1 * kernel_size_0 * kernel_size_0 + 8 * kernel_size_0 * kernel_size_1
        index_mask = combined_index < divisor

        input_value = tl.load(
            input_ptr + (2 * (((combined_index // (2 + kernel_size_1)) % (2 + kernel_size_1)) + 
                              4 * ((combined_index // (4 + kernel_size_1 * kernel_size_1 + 4 * kernel_size_1)) % (2 + kernel_size_0)) + 
                              8 * x_mod_32 + 256 * ((combined_index // (8 + 2 * kernel_size_1 * kernel_size_1 + 4 * kernel_size_0 + 
                                                                        8 * kernel_size_1 + kernel_size_0 * kernel_size_1 * kernel_size_1 + 
                                                                        4 * kernel_size_0 * kernel_size_1)) % kernel_size_0) + 
                              kernel_size_1 * (((combined_index // (2 + kernel_size_1)) % (2 + kernel_size_1))) + 
                              kernel_size_1 * kernel_size_1 * ((combined_index // (4 + kernel_size_1 * kernel_size_1 + 4 * kernel_size_1)) % (2 + kernel_size_0)) + 
                              2 * x_mod_32 * kernel_size_1 * kernel_size_1 + 4 * kernel_size_0 * x_mod_32 + 
                              4 * kernel_size_1 * ((combined_index // (4 + kernel_size_1 * kernel_size_1 + 4 * kernel_size_1)) % (2 + kernel_size_0)) + 
                              8 * kernel_size_1 * x_mod_32 + 64 * kernel_size_1 * kernel_size_1 * ((combined_index // (8 + 2 * kernel_size_1 * kernel_size_1 + 4 * kernel_size_0 + 
                                                                                                              8 * kernel_size_1 + kernel_size_0 * kernel_size_1 * kernel_size_1 + 
                                                                                                              4 * kernel_size_0 * kernel_size_1)) % kernel_size_0) + 
                              128 * kernel_size_0 * ((combined_index // (8 + 2 * kernel_size_1 * kernel_size_1 + 4 * kernel_size_0 + 
                                                                        8 * kernel_size_1 + kernel_size_0 * kernel_size_1 * kernel_size_1 + 
                                                                        4 * kernel_size_0 * kernel_size_1)) % kernel_size_0) + 
                              256 * kernel_size_1 * ((combined_index // (8 + 2 * kernel_size_1 * kernel_size_1 + 4 * kernel_size_0 + 
                                                                        8 * kernel_size_1 + kernel_size_0 * kernel_size_1 * kernel_size_1 + 
                                                                        4 * kernel_size_0 * kernel_size_1)) % kernel_size_0) + 
                              kernel_size_0 * x_mod_32 * kernel_size_1 * kernel_size_1 + 4 * kernel_size_0 * kernel_size_1 * x_mod_32 + 
                              32 * kernel_size_0 * kernel_size_1 * kernel_size_1 * ((combined_index // (8 + 2 * kernel_size_1 * kernel_size_1 + 4 * kernel_size_0 + 
                                                                                                              8 * kernel_size_1 + kernel_size_0 * kernel_size_1 * kernel_size_1 + 
                                                                                                              4 * kernel_size_0 * kernel_size_1)) % kernel_size_0) + 
                              128 * kernel_size_0 * kernel_size_1 * ((combined_index // (8 + 2 * kernel_size_1 * kernel_size_1 + 4 * kernel_size_0 + 
                                                                                                           8 * kernel_size_1 + kernel_size_0 * kernel_size_1 * kernel_size_1 + 
                                                                                                           4 * kernel_size_0 * kernel_size_1)) % kernel_size_0) + 
                              (combined_index % (2 + kernel_size_1)))), r_mask & index_mask & x_mask, 
            eviction_policy='evict_last', other=0.0
        )

        scale_factor = 2.0
        scaled_input = input_value * scale_factor
        zero_tensor = tl.full(scaled_input.shape, 0, scaled_input.dtype)
        masked_input = tl.where(index_mask, scaled_input, zero_tensor)
        zero_value = 0.0
        zero_broadcast = tl.full(zero_value.shape, 0, zero_value.dtype)
        masked_zero = tl.where(index_mask, zero_value, zero_broadcast)
        one_value = 1.0
        one_broadcast = tl.full(one_value.shape, 0, one_value.dtype)
        masked_one = tl.where(index_mask, one_value, one_broadcast)

        broadcasted_input = tl.broadcast_to(masked_input, [XBLOCK, RBLOCK])
        broadcasted_zero = tl.broadcast_to(masked_zero, [XBLOCK, RBLOCK])
        broadcasted_one = tl.broadcast_to(masked_one, [XBLOCK, RBLOCK])

        tmp_mean_next, tmp_m2_next, tmp_weight_next = triton_helpers.welford_combine(
            tmp_mean, tmp_m2, tmp_weight,
            broadcasted_input, broadcasted_zero, broadcasted_one
        )

        tmp_mean = tl.where(r_mask & x_mask, tmp_mean_next, tmp_mean)
        tmp_m2 = tl.where(r_mask & x_mask, tmp_m2_next, tmp_m2)
        tmp_weight = tl.where(r_mask & x_mask, tmp_weight_next, tmp_weight)

    mean_result, variance_result, weight_result = triton_helpers.welford(
        tmp_mean, tmp_m2, tmp_weight, 1
    )

    mean_result = mean_result[:, None]
    variance_result = variance_result[:, None]
    weight_result = weight_result[:, None]

    tl.store(output_mean_ptr + (x3), mean_result, x_mask)
    tl.store(output_var_ptr + (x3), variance_result, x_mask)
    tl.store(output_weight_ptr + (x3), weight_result, x_mask)