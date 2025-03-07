# From: 60_ConvTranspose3d_Swish_GroupNorm_HardSwish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_group_norm_backward_2(
    input_grad_ptr, input_ptr, mean_ptr, inv_std_ptr, output_grad_ptr, output_ptr, 
    xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    xnumel = 16
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x3 = x_index
    x1 = x_index // 4
    temp_sum_grad = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    temp_sum_input = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r2 = r_index
        grad_input = tl.load(input_grad_ptr + (x3 + 16 * r2), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        input_data = tl.load(input_ptr + (x3 + 16 * r2), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        mean = tl.load(mean_ptr + (x1 + 4 * r2), r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        inv_std = tl.load(inv_std_ptr + (x1 + 4 * r2), r_mask & x_mask, eviction_policy='evict_last', other=0.0)

        input_mean = input_data * mean
        grad_input_centered = grad_input - input_mean
        grad_input_scaled = grad_input_centered * inv_std
        grad_input_scaled_broadcast = tl.broadcast_to(grad_input_scaled, [XBLOCK, RBLOCK])

        temp_sum_grad += grad_input_scaled_broadcast
        temp_sum_grad = tl.where(r_mask & x_mask, temp_sum_grad, temp_sum_grad)

        input_data_broadcast = tl.broadcast_to(input_data, [XBLOCK, RBLOCK])
        temp_sum_input += input_data_broadcast
        temp_sum_input = tl.where(r_mask & x_mask, temp_sum_input, temp_sum_input)

    sum_grad = tl.sum(temp_sum_grad, 1)[:, None]
    sum_input = tl.sum(temp_sum_input, 1)[:, None]

    tl.store(output_grad_ptr + (x3), sum_grad, x_mask)
    tl.store(output_ptr + (x3), sum_input, x_mask)