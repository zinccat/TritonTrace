# From: 60_ConvTranspose3d_Swish_GroupNorm_HardSwish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_group_norm_backward_1red_fused_native_group_norm_backward_1(
    input_grad_ptr, input_data_ptr, input_scale_ptr, output_grad_ptr0, output_grad_ptr1, 
    kernel_size0, kernel_size1, kernel_size2, input_num_elements, reduction_num_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < input_num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_indices_0 = x_indices
    temp_sum0 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    temp_sum1 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for r_offset in range(0, reduction_num_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < reduction_num_elements
        r_indices_1 = r_indices

        index0 = (
            (-1) * x_indices_0 
            + (-1) * ((r_indices_1 // kernel_size0) % kernel_size0) 
            + (-4) * kernel_size2 * (triton_helpers.div_floor_integer(r_indices_1, 1 + (-4) * kernel_size2 + 4 * kernel_size2 * kernel_size2)) 
            + (-4) * x_indices_0 * kernel_size2 * kernel_size2 
            + 2 * kernel_size1 * x_indices_0 
            + 2 * kernel_size2 * ((r_indices_1 // kernel_size0) % kernel_size0) 
            + 4 * kernel_size2 * x_indices_0 
            + 4 * kernel_size2 * kernel_size2 * (triton_helpers.div_floor_integer(r_indices_1, 1 + (-4) * kernel_size2 + 4 * kernel_size2 * kernel_size2)) 
            + (-8) * kernel_size1 * kernel_size2 * x_indices_0 
            + 8 * kernel_size1 * x_indices_0 * kernel_size2 * kernel_size2 
            + (triton_helpers.div_floor_integer(r_indices_1, 1 + (-4) * kernel_size2 + 4 * kernel_size2 * kernel_size2)) 
            + (r_indices_1 % kernel_size0)
        )

        grad_input = tl.load(input_grad_ptr + index0, r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        grad_data = tl.load(input_data_ptr + index0, r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        scale = tl.load(input_scale_ptr + index0, r_mask & x_mask, eviction_policy='evict_last', other=0.0)

        threshold_neg = -3.0
        mask_neg = grad_input < threshold_neg
        threshold_pos = 3.0
        mask_pos = grad_input <= threshold_pos

        scale_factor = 0.3333333333333333
        scaled_input = grad_input * scale_factor
        bias = 0.5
        adjusted_input = scaled_input + bias
        adjusted_grad = grad_data * adjusted_input

        clamped_grad = tl.where(mask_pos, adjusted_grad, grad_data)
        zero_grad = 0.0
        final_grad = tl.where(mask_neg, zero_grad, clamped_grad)

        sigmoid_scale = tl.sigmoid(scale)
        scaled_sigmoid = sigmoid_scale * scale
        weighted_grad = final_grad * scaled_sigmoid

        broadcast_weighted_grad = tl.broadcast_to(weighted_grad, [XBLOCK, RBLOCK])
        temp_sum0 = temp_sum0 + broadcast_weighted_grad
        temp_sum0 = tl.where(r_mask & x_mask, temp_sum0, temp_sum0)

        broadcast_final_grad = tl.broadcast_to(final_grad, [XBLOCK, RBLOCK])
        temp_sum1 = temp_sum1 + broadcast_final_grad
        temp_sum1 = tl.where(r_mask & x_mask, temp_sum1, temp_sum1)

    sum_temp0 = tl.sum(temp_sum0, 1)[:, None]
    sum_temp1 = tl.sum(temp_sum1, 1)[:, None]

    tl.store(output_grad_ptr0 + (x_indices_0), sum_temp0, x_mask)
    tl.store(output_grad_ptr1 + (x_indices_0), sum_temp1, x_mask)