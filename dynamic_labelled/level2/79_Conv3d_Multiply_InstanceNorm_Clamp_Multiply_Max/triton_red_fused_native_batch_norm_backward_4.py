# From: 79_Conv3d_Multiply_InstanceNorm_Clamp_Multiply_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_batch_norm_backward_4(
    input_grad_ptr, input_ptr, input_multiplier_ptr, input_clamp_ptr, 
    output_grad_ptr, output_multiplier_ptr, kernel_size_0, kernel_size_1, 
    input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, 
    RBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < input_num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_indices_flat = x_indices
    temp_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_multiplier = tl.load(input_multiplier_ptr + ((x_indices_flat % 16)), x_mask, eviction_policy='evict_last')
    input_clamp = tl.load(input_clamp_ptr + (x_indices_flat), x_mask, eviction_policy='evict_last')
    temp_product = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for r_offset in range(0, reduction_num_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < reduction_num_elements
        r_indices_flat = r_indices

        input_grad = tl.load(
            input_grad_ptr + (r_indices_flat + ((-8) * x_indices_flat) + ((-2) * x_indices_flat * kernel_size_1 * kernel_size_1) + 
                              4 * kernel_size_0 * x_indices_flat + 8 * kernel_size_1 * x_indices_flat + 
                              kernel_size_0 * x_indices_flat * kernel_size_1 * kernel_size_1 + ((-4) * kernel_size_0 * kernel_size_1 * x_indices_flat)),
            r_mask & x_mask, eviction_policy='evict_first', other=0.0
        )

        input_multiplier_grad = tl.load(
            input_multiplier_ptr + (r_indices_flat + ((-8) * x_indices_flat) + ((-2) * x_indices_flat * kernel_size_1 * kernel_size_1) + 
                                    4 * kernel_size_0 * x_indices_flat + 8 * kernel_size_1 * x_indices_flat + 
                                    kernel_size_0 * x_indices_flat * kernel_size_1 * kernel_size_1 + ((-4) * kernel_size_0 * kernel_size_1 * x_indices_flat)),
            r_mask & x_mask, eviction_policy='evict_first', other=0.0
        )

        broadcast_input_grad = tl.broadcast_to(input_grad, [XBLOCK, RBLOCK])
        temp_sum_update = temp_sum + broadcast_input_grad
        temp_sum = tl.where(r_mask & x_mask, temp_sum_update, temp_sum)

        input_multiplier_grad_product = input_multiplier_grad * input_multiplier
        input_clamp_difference = input_multiplier_grad_product - input_clamp
        input_grad_product = input_grad * input_clamp_difference
        broadcast_input_grad_product = tl.broadcast_to(input_grad_product, [XBLOCK, RBLOCK])
        temp_product_update = temp_product + broadcast_input_grad_product
        temp_product = tl.where(r_mask & x_mask, temp_product_update, temp_product)

    output_grad_sum = tl.sum(temp_sum, 1)[:, None]
    output_multiplier_sum = tl.sum(temp_product, 1)[:, None]
    tl.store(output_grad_ptr + (x_indices_flat), output_grad_sum, x_mask)
    tl.store(output_multiplier_ptr + (x_indices_flat), output_multiplier_sum, x_mask)