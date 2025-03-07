# From: 60_ConvTranspose3d_Swish_GroupNorm_HardSwish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_group_norm_backward_1(
    input_grad_ptr, input_ptr, weight_ptr, output_grad_ptr0, output_grad_ptr1, 
    kernel_size0, kernel_size1, kernel_size2, input_num_elements, reduction_num_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < input_num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_indices_flat = x_indices
    temp_sum_grad = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    temp_sum_input = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for r_offset in range(0, reduction_num_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < reduction_num_elements
        r_indices_flat = r_indices

        grad_index = (
            (-1) * x_indices_flat
            + (-1) * ((r_indices_flat // kernel_size0) % kernel_size0)
            + (-4) * kernel_size2 * (triton_helpers.div_floor_integer(r_indices_flat, 1 + (-4) * kernel_size2 + 4 * kernel_size2 * kernel_size2))
            + (-4) * x_indices_flat * kernel_size2 * kernel_size2
            + 2 * kernel_size1 * x_indices_flat
            + 2 * kernel_size2 * ((r_indices_flat // kernel_size0) % kernel_size0)
            + 4 * kernel_size2 * x_indices_flat
            + 4 * kernel_size2 * kernel_size2 * (triton_helpers.div_floor_integer(r_indices_flat, 1 + (-4) * kernel_size2 + 4 * kernel_size2 * kernel_size2))
            + (-8) * kernel_size1 * kernel_size2 * x_indices_flat
            + 8 * kernel_size1 * x_indices_flat * kernel_size2 * kernel_size2
            + (triton_helpers.div_floor_integer(r_indices_flat, 1 + (-4) * kernel_size2 + 4 * kernel_size2 * kernel_size2))
            + (r_indices_flat % kernel_size0)
        )

        input_grad = tl.load(input_grad_ptr + grad_index, r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        input_value = tl.load(input_ptr + grad_index, r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        weight_value = tl.load(weight_ptr + grad_index, r_mask & x_mask, eviction_policy='evict_last', other=0.0)

        threshold_neg = -3.0
        threshold_pos = 3.0
        scale_factor = 0.3333333333333333
        bias = 0.5

        scaled_input = input_grad * scale_factor
        biased_input = scaled_input + bias
        adjusted_input = tl.where(input_grad <= threshold_pos, input_value * biased_input, input_value)

        clamped_input = tl.where(input_grad < threshold_neg, 0.0, adjusted_input)

        sigmoid_weight = tl.sigmoid(weight_value)
        activated_weight = sigmoid_weight * weight_value

        grad_contribution = clamped_input * activated_weight
        broadcast_grad_contribution = tl.broadcast_to(grad_contribution, [XBLOCK, RBLOCK])

        temp_sum_grad += tl.where(r_mask & x_mask, broadcast_grad_contribution, temp_sum_grad)
        temp_sum_input += tl.where(r_mask & x_mask, tl.broadcast_to(clamped_input, [XBLOCK, RBLOCK]), temp_sum_input)

    summed_grad = tl.sum(temp_sum_grad, 1)[:, None]
    summed_input = tl.sum(temp_sum_input, 1)[:, None]

    tl.store(output_grad_ptr0 + x_indices_flat, summed_grad, x_mask)
    tl.store(output_grad_ptr1 + x_indices_flat, summed_input, x_mask)