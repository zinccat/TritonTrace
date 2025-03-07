# From: 11_ConvTranspose2d_BatchNorm_Tanh_MaxPool_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_batch_norm_backward_tanh_backward_5(
    input_grad_ptr, input_ptr, running_mean_ptr, running_var_ptr, 
    output_grad_ptr0, output_grad_ptr1, kernel_size, 
    xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    xnumel = 384
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_channel = x_index // 64
    x_pixel = x_index % 64
    temp_sum0 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x_flat_index = x_index
    temp_sum1 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r_flat_index = r_index
        temp_index = r_flat_index + x_channel * ((5 + 4096 * kernel_size) // 6)
        max_index = 4096 * kernel_size
        valid_mask = temp_index < max_index

        input_grad = tl.load(
            input_grad_ptr + (4096 * x_pixel + 262144 * (((temp_index // 4096) % kernel_size) + (temp_index % 4096))),
            valid_mask & x_mask, eviction_policy='evict_last', other=0.0
        )
        input_value = tl.load(
            input_ptr + (4096 * x_pixel + 262144 * (((temp_index // 4096) % kernel_size) + (temp_index % 4096))),
            valid_mask & x_mask, eviction_policy='evict_last', other=0.0
        )
        input_squared = input_value * input_value
        variance = 1.0 - input_squared
        normalized_grad = input_grad * variance

        temp_grad = tl.full(normalized_grad.shape, 0, normalized_grad.dtype)
        valid_grad = tl.where(valid_mask, normalized_grad, temp_grad)
        broadcast_grad = tl.broadcast_to(valid_grad, [XBLOCK, RBLOCK])
        temp_sum0 += broadcast_grad
        temp_sum0 = tl.where(r_mask & x_mask, temp_sum0, temp_sum0)

        running_mean = tl.load(
            running_mean_ptr + (4096 * x_pixel + 262144 * (((temp_index // 4096) % kernel_size) + (temp_index % 4096))),
            valid_mask & x_mask, eviction_policy='evict_last', other=0.0
        )
        running_var = tl.load(
            running_var_ptr + (tl.broadcast_to(x_pixel, [XBLOCK, RBLOCK])),
            valid_mask & x_mask, eviction_policy='evict_last', other=0.0
        )
        mean_diff = running_mean - running_var
        grad_diff = normalized_grad * mean_diff

        temp_diff = tl.full(grad_diff.shape, 0, grad_diff.dtype)
        valid_diff = tl.where(valid_mask, grad_diff, temp_diff)
        broadcast_diff = tl.broadcast_to(valid_diff, [XBLOCK, RBLOCK])
        temp_sum1 += broadcast_diff
        temp_sum1 = tl.where(r_mask & x_mask, temp_sum1, temp_sum1)

    output_grad0 = tl.sum(temp_sum0, 1)[:, None]
    output_grad1 = tl.sum(temp_sum1, 1)[:, None]
    tl.store(output_grad_ptr0 + (x_flat_index), output_grad0, x_mask)
    tl.store(output_grad_ptr1 + (x_flat_index), output_grad1, x_mask)