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
    temp_sum_multiplier = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    input_multiplier = tl.load(input_multiplier_ptr + ((x_indices_flat % 16)), x_mask, eviction_policy='evict_last')
    input_clamp = tl.load(input_clamp_ptr + (x_indices_flat), x_mask, eviction_policy='evict_last')
    temp_sum_grad = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for r_offset in range(0, reduction_num_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < reduction_num_elements
        r_indices_flat = r_indices

        grad_index = (
            r_indices_flat + 
            ((-8) * x_indices_flat) + 
            ((-2) * x_indices_flat * kernel_size_1 * kernel_size_1) + 
            4 * kernel_size_0 * x_indices_flat + 
            8 * kernel_size_1 * x_indices_flat + 
            kernel_size_0 * x_indices_flat * kernel_size_1 * kernel_size_1 + 
            ((-4) * kernel_size_0 * kernel_size_1 * x_indices_flat)
        )
        
        input_grad = tl.load(input_grad_ptr + grad_index, r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        input_multiplier_grad = tl.load(input_multiplier_ptr + grad_index, r_mask & x_mask, eviction_policy='evict_first', other=0.0)

        broadcast_input_grad = tl.broadcast_to(input_grad, [XBLOCK, RBLOCK])
        temp_sum_multiplier = temp_sum_multiplier + broadcast_input_grad
        temp_sum_multiplier = tl.where(r_mask & x_mask, temp_sum_multiplier, temp_sum_multiplier)

        multiplier_diff = input_multiplier_grad * input_multiplier
        clamp_diff = multiplier_diff - input_clamp
        grad_contribution = input_grad * clamp_diff
        broadcast_grad_contribution = tl.broadcast_to(grad_contribution, [XBLOCK, RBLOCK])
        temp_sum_grad = temp_sum_grad + broadcast_grad_contribution
        temp_sum_grad = tl.where(r_mask & x_mask, temp_sum_grad, temp_sum_grad)

    summed_temp_sum_multiplier = tl.sum(temp_sum_multiplier, 1)[:, None]
    summed_temp_sum_grad = tl.sum(temp_sum_grad, 1)[:, None]

    tl.store(output_grad_ptr + (x_indices_flat), summed_temp_sum_multiplier, x_mask)
    tl.store(output_multiplier_ptr + (x_indices_flat), summed_temp_sum_grad, x_mask)