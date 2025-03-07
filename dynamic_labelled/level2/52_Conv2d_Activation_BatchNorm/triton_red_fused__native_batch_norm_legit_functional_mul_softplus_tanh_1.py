# From: 52_Conv2d_Activation_BatchNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_mul_softplus_tanh_1red_fused__native_batch_norm_legit_functional_mul_softplus_tanh_1(
    input_ptr, output_ptr_mean, output_ptr_var, output_ptr_count, kernel_size_0, kernel_size_1, total_elements, reduction_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    total_elements = 240
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < total_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_channel = x_indices // 16
    x_within_channel = x_indices % 16
    temp_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x_flat_index = x_indices

    for r_offset in range(0, reduction_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < reduction_elements
        r_flat_index = r_indices
        temp_index = r_flat_index + x_channel * (
            triton_helpers.div_floor_integer(14 + 4 * kernel_size_0 + kernel_size_0 * kernel_size_1 * kernel_size_1 + (-4) * kernel_size_0 * kernel_size_1, 15)
        )
        temp_limit = 4 * kernel_size_0 + kernel_size_0 * kernel_size_1 * kernel_size_1 + (-4) * kernel_size_0 * kernel_size_1
        valid_index_mask = temp_index < temp_limit
        input_value = tl.load(
            input_ptr + (
                -2 * (
                    (
                        (r_flat_index + x_channel * (
                            triton_helpers.div_floor_integer(14 + 4 * kernel_size_0 + kernel_size_0 * kernel_size_1 * kernel_size_1 + (-4) * kernel_size_0 * kernel_size_1, 15)
                        )) // (-2 + kernel_size_1)
                    ) % (-2 + kernel_size_1)
                ) + 4 * x_within_channel + 64 * (
                    (
                        (r_flat_index + x_channel * (
                            triton_helpers.div_floor_integer(14 + 4 * kernel_size_0 + kernel_size_0 * kernel_size_1 * kernel_size_1 + (-4) * kernel_size_0 * kernel_size_1, 15)
                        )) // (4 + kernel_size_1 * kernel_size_1 + (-4) * kernel_size_1)
                    ) % kernel_size_0
                ) + kernel_size_1 * (
                    (
                        (r_flat_index + x_channel * (
                            triton_helpers.div_floor_integer(14 + 4 * kernel_size_0 + kernel_size_0 * kernel_size_1 * kernel_size_1 + (-4) * kernel_size_0 * kernel_size_1, 15)
                        )) // (-2 + kernel_size_1)
                    ) % (-2 + kernel_size_1)
                ) + x_within_channel * kernel_size_1 * kernel_size_1 + (-64) * kernel_size_1 * (
                    (
                        (r_flat_index + x_channel * (
                            triton_helpers.div_floor_integer(14 + 4 * kernel_size_0 + kernel_size_0 * kernel_size_1 * kernel_size_1 + (-4) * kernel_size_0 * kernel_size_1, 15)
                        )) // (4 + kernel_size_1 * kernel_size_1 + (-4) * kernel_size_1)
                    ) % kernel_size_0
                ) + (-4) * kernel_size_1 * x_within_channel + 16 * kernel_size_1 * kernel_size_1 * (
                    (
                        (r_flat_index + x_channel * (
                            triton_helpers.div_floor_integer(14 + 4 * kernel_size_0 + kernel_size_0 * kernel_size_1 * kernel_size_1 + (-4) * kernel_size_0 * kernel_size_1, 15)
                        )) // (4 + kernel_size_1 * kernel_size_1 + (-4) * kernel_size_1)
                    ) % kernel_size_0
                ) + (r_flat_index + x_channel * (
                    triton_helpers.div_floor_integer(14 + 4 * kernel_size_0 + kernel_size_0 * kernel_size_1 * kernel_size_1 + (-4) * kernel_size_0 * kernel_size_1, 15)
                )) % (-2 + kernel_size_1)
            ),
            r_mask & valid_index_mask & x_mask,
            eviction_policy='evict_last',
            other=0.0
        )
        threshold = 20.0
        is_greater_than_threshold = input_value > threshold
        exp_value = tl.math.exp(input_value)
        log1p_exp_value = tl.extra.cuda.libdevice.log1p(exp_value)
        softplus_value = tl.where(is_greater_than_threshold, input_value, log1p_exp_value)
        tanh_value = tl.extra.cuda.libdevice.tanh(softplus_value)
        weighted_input = tanh_value * input_value
        zero_tensor = tl.full(weighted_input.shape, 0, weighted_input.dtype)
        masked_weighted_input = tl.where(valid_index_mask, weighted_input, zero_tensor)
        zero_value = 0.0
        zero_broadcast = tl.full(zero_value.shape, 0, zero_value.dtype)
        masked_zero = tl.where(valid_index_mask, zero_value, zero_broadcast)
        one_value = 1.0
        one_broadcast = tl.full(one_value.shape, 0, one_value.dtype)
        masked_one = tl.where(valid_index_mask, one_value, one_broadcast)
        broadcast_weighted_input = tl.broadcast_to(masked_weighted_input, [XBLOCK, RBLOCK])
        broadcast_zero = tl.broadcast_to(masked_zero, [XBLOCK, RBLOCK])
        broadcast_one = tl.broadcast_to(masked_one, [XBLOCK, RBLOCK])
        temp_mean_next, temp_m2_next, temp_weight_next = triton_helpers.welford_combine(
            temp_mean, temp_m2, temp_weight,
            broadcast_weighted_input, broadcast_zero, broadcast_one
        )
        temp_mean = tl.where(r_mask & x_mask, temp_mean_next, temp_mean)
        temp_m2 = tl.where(r_mask & x_mask, temp_m2_next, temp_m2)
        temp_weight = tl.where(r_mask & x_mask, temp_weight_next, temp_weight)

    mean_result, variance_result, count_result = triton_helpers.welford(
        temp_mean, temp_m2, temp_weight, 1
    )
    mean_result = mean_result[:, None]
    variance_result = variance_result[:, None]
    count_result = count_result[:, None]
    tl.store(output_ptr_mean + (x_flat_index), mean_result, x_mask)
    tl.store(output_ptr_var + (x_flat_index), variance_result, x_mask)
    tl.store(output_ptr_count + (x_flat_index), count_result, x_mask)