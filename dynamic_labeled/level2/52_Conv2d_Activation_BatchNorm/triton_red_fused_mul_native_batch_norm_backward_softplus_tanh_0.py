# From: 52_Conv2d_Activation_BatchNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_mul_native_batch_norm_backward_softplus_tanh_0(
    input_grad_ptr, input_ptr, weight_ptr, output_grad_ptr0, output_grad_ptr1, kernel_size0, kernel_size1, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    input_num_elements = 240
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_index_16 = input_index // 16
    input_index_mod_16 = input_index % 16
    temp_sum0 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_index_flat = input_index
    temp_sum1 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_index_flat = reduction_index
        temp_index = reduction_index_flat + input_index_16 * (
            triton_helpers.div_floor_integer(14 + 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + (-4) * kernel_size0 * kernel_size1, 15)
        )
        temp_limit = 4 * kernel_size0 + kernel_size0 * kernel_size1 * kernel_size1 + (-4) * kernel_size0 * kernel_size1
        temp_condition = temp_index < temp_limit

        input_grad = tl.load(
            input_grad_ptr + (
                (-2) * (((temp_index // ((-2) + kernel_size1)) % ((-2) + kernel_size1))) +
                4 * input_index_mod_16 +
                64 * ((temp_index // (4 + kernel_size1 * kernel_size1 + (-4) * kernel_size1)) % kernel_size0) +
                kernel_size1 * ((temp_index // ((-2) + kernel_size1)) % ((-2) + kernel_size1)) +
                input_index_mod_16 * kernel_size1 * kernel_size1 +
                (-64) * kernel_size1 * ((temp_index // (4 + kernel_size1 * kernel_size1 + (-4) * kernel_size1)) % kernel_size0) +
                (-4) * kernel_size1 * input_index_mod_16 +
                16 * kernel_size1 * kernel_size1 * ((temp_index // (4 + kernel_size1 * kernel_size1 + (-4) * kernel_size1)) % kernel_size0) +
                (temp_index % ((-2) + kernel_size1))
            ),
            reduction_mask & temp_condition & input_mask,
            eviction_policy='evict_last',
            other=0.0
        )

        broadcast_input_grad = tl.broadcast_to(input_grad, [XBLOCK, RBLOCK])
        temp_sum0_updated = temp_sum0 + broadcast_input_grad
        temp_sum0 = tl.where(reduction_mask & input_mask, temp_sum0_updated, temp_sum0)

        input_value = tl.load(
            input_ptr + (
                (-2) * (((temp_index // ((-2) + kernel_size1)) % ((-2) + kernel_size1))) +
                4 * input_index_mod_16 +
                64 * ((temp_index // (4 + kernel_size1 * kernel_size1 + (-4) * kernel_size1)) % kernel_size0) +
                kernel_size1 * ((temp_index // ((-2) + kernel_size1)) % ((-2) + kernel_size1)) +
                input_index_mod_16 * kernel_size1 * kernel_size1 +
                (-64) * kernel_size1 * ((temp_index // (4 + kernel_size1 * kernel_size1 + (-4) * kernel_size1)) % kernel_size0) +
                (-4) * kernel_size1 * input_index_mod_16 +
                16 * kernel_size1 * kernel_size1 * ((temp_index // (4 + kernel_size1 * kernel_size1 + (-4) * kernel_size1)) % kernel_size0) +
                (temp_index % ((-2) + kernel_size1))
            ),
            reduction_mask & temp_condition & input_mask,
            eviction_policy='evict_last',
            other=0.0
        )

        threshold = 20.0
        is_greater_than_threshold = input_value > threshold
        exp_input_value = tl.math.exp(input_value)
        log1p_exp_input_value = tl.extra.cuda.libdevice.log1p(exp_input_value)
        adjusted_input_value = tl.where(is_greater_than_threshold, input_value, log1p_exp_input_value)
        tanh_adjusted_value = tl.extra.cuda.libdevice.tanh(adjusted_input_value)
        product_tanh_input = tanh_adjusted_value * input_value

        weight_value = tl.load(
            weight_ptr + (tl.broadcast_to(input_index_mod_16, [XBLOCK, RBLOCK])),
            reduction_mask & temp_condition & input_mask,
            eviction_policy='evict_last',
            other=0.0
        )

        difference = product_tanh_input - weight_value
        product_input_grad_difference = input_grad * difference
        zero_filled_product = tl.full(product_input_grad_difference.shape, 0, product_input_grad_difference.dtype)
        conditional_product = tl.where(temp_condition, product_input_grad_difference, zero_filled_product)
        broadcast_conditional_product = tl.broadcast_to(conditional_product, [XBLOCK, RBLOCK])
        temp_sum1_updated = temp_sum1 + broadcast_conditional_product
        temp_sum1 = tl.where(reduction_mask & input_mask, temp_sum1_updated, temp_sum1)

    output_grad0 = tl.sum(temp_sum0, 1)[:, None]
    output_grad1 = tl.sum(temp_sum1, 1)[:, None]
    tl.store(output_grad_ptr0 + (input_index_flat), output_grad0, input_mask)
    tl.store(output_grad_ptr1 + (input_index_flat), output_grad1, input_mask)