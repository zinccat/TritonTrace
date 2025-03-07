# From: 52_Conv2d_Activation_BatchNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_mul_softplus_tanh_1(
    input_ptr, output_mean_ptr, output_var_ptr, output_weight_ptr, kernel_size_0, kernel_size_1, 
    input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    input_num_elements = 240
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_index_1 = input_index // 16
    input_index_0 = (input_index % 16)
    temp_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    input_index_3 = input_index

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_index_2 = reduction_index
        temp_index_0 = reduction_index_2 + input_index_1 * (
            triton_helpers.div_floor_integer(14 + 4 * kernel_size_0 + kernel_size_0 * kernel_size_1 * kernel_size_1 + (-4) * kernel_size_0 * kernel_size_1, 15)
        )
        temp_index_1 = 4 * kernel_size_0 + kernel_size_0 * kernel_size_1 * kernel_size_1 + (-4) * kernel_size_0 * kernel_size_1
        temp_mask_0 = temp_index_0 < temp_index_1
        temp_value = tl.load(
            input_ptr + (
                -2 * (
                    (
                        (reduction_index_2 + input_index_1 * (
                            triton_helpers.div_floor_integer(14 + 4 * kernel_size_0 + kernel_size_0 * kernel_size_1 * kernel_size_1 + (-4) * kernel_size_0 * kernel_size_1, 15)
                        )) // (-2 + kernel_size_1)
                    ) % (-2 + kernel_size_1)
                ) + 4 * input_index_0 + 64 * (
                    (
                        (reduction_index_2 + input_index_1 * (
                            triton_helpers.div_floor_integer(14 + 4 * kernel_size_0 + kernel_size_0 * kernel_size_1 * kernel_size_1 + (-4) * kernel_size_0 * kernel_size_1, 15)
                        )) // (4 + kernel_size_1 * kernel_size_1 + (-4) * kernel_size_1)
                    ) % kernel_size_0
                ) + kernel_size_1 * (
                    (
                        (reduction_index_2 + input_index_1 * (
                            triton_helpers.div_floor_integer(14 + 4 * kernel_size_0 + kernel_size_0 * kernel_size_1 * kernel_size_1 + (-4) * kernel_size_0 * kernel_size_1, 15)
                        )) // (-2 + kernel_size_1)
                    ) % (-2 + kernel_size_1)
                ) + input_index_0 * kernel_size_1 * kernel_size_1 + (-64) * kernel_size_1 * (
                    (
                        (reduction_index_2 + input_index_1 * (
                            triton_helpers.div_floor_integer(14 + 4 * kernel_size_0 + kernel_size_0 * kernel_size_1 * kernel_size_1 + (-4) * kernel_size_0 * kernel_size_1, 15)
                        )) // (4 + kernel_size_1 * kernel_size_1 + (-4) * kernel_size_1)
                    ) % kernel_size_0
                ) + (-4) * kernel_size_1 * input_index_0 + 16 * kernel_size_1 * kernel_size_1 * (
                    (
                        (reduction_index_2 + input_index_1 * (
                            triton_helpers.div_floor_integer(14 + 4 * kernel_size_0 + kernel_size_0 * kernel_size_1 * kernel_size_1 + (-4) * kernel_size_0 * kernel_size_1, 15)
                        )) // (4 + kernel_size_1 * kernel_size_1 + (-4) * kernel_size_1)
                    ) % kernel_size_0
                ) + (
                    (reduction_index_2 + input_index_1 * (
                        triton_helpers.div_floor_integer(14 + 4 * kernel_size_0 + kernel_size_0 * kernel_size_1 * kernel_size_1 + (-4) * kernel_size_0 * kernel_size_1, 15)
                    )) % (-2 + kernel_size_1)
                )
            ),
            reduction_mask & temp_mask_0 & input_mask,
            eviction_policy='evict_last',
            other=0.0
        )
        threshold_value = 20.0
        is_greater_than_threshold = temp_value > threshold_value
        exp_value = tl.math.exp(temp_value)
        log1p_value = tl.extra.cuda.libdevice.log1p(exp_value)
        adjusted_value = tl.where(is_greater_than_threshold, temp_value, log1p_value)
        tanh_value = tl.extra.cuda.libdevice.tanh(adjusted_value)
        product_value = tanh_value * temp_value
        zero_value = tl.full(product_value.shape, 0, product_value.dtype)
        masked_product_value = tl.where(temp_mask_0, product_value, zero_value)
        zero_broadcast = tl.full(zero_value.shape, 0, zero_value.dtype)
        one_broadcast = tl.full(zero_value.shape, 1, zero_value.dtype)
        masked_zero_broadcast = tl.where(temp_mask_0, zero_value, zero_broadcast)
        masked_one_broadcast = tl.where(temp_mask_0, one_broadcast, one_broadcast)
        broadcasted_product = tl.broadcast_to(masked_product_value, [XBLOCK, RBLOCK])
        broadcasted_zero = tl.broadcast_to(masked_zero_broadcast, [XBLOCK, RBLOCK])
        broadcasted_one = tl.broadcast_to(masked_one_broadcast, [XBLOCK, RBLOCK])
        temp_mean_next, temp_m2_next, temp_weight_next = triton_helpers.welford_combine(
            temp_mean, temp_m2, temp_weight,
            broadcasted_product, broadcasted_zero, broadcasted_one
        )
        temp_mean = tl.where(reduction_mask & input_mask, temp_mean_next, temp_mean)
        temp_m2 = tl.where(reduction_mask & input_mask, temp_m2_next, temp_m2)
        temp_weight = tl.where(reduction_mask & input_mask, temp_weight_next, temp_weight)

    mean_result, variance_result, weight_result = triton_helpers.welford(
        temp_mean, temp_m2, temp_weight, 1
    )
    mean_result_broadcast = mean_result[:, None]
    variance_result_broadcast = variance_result[:, None]
    weight_result_broadcast = weight_result[:, None]
    tl.store(output_mean_ptr + (input_index_3), mean_result_broadcast, input_mask)
    tl.store(output_var_ptr + (input_index_3), variance_result_broadcast, input_mask)
    tl.store(output_weight_ptr + (input_index_3), weight_result_broadcast, input_mask)