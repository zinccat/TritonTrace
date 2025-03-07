# From: 17_Conv2d_InstanceNorm_Divide

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_batch_norm_backward_0(
    input_grad_ptr, mean_ptr, variance_ptr, output_grad_mean_ptr, output_grad_variance_ptr,
    kernel_size, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < input_num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_indices_flat = x_indices
    temp_sum_mean = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    variance_values = tl.load(variance_ptr + (x_indices_flat), x_mask, eviction_policy='evict_last')
    temp_sum_variance = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for r_offset in range(0, reduction_num_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < reduction_num_elements
        r_indices_flat = r_indices
        grad_input_values = tl.load(
            input_grad_ptr + (r_indices_flat + 4 * x_indices_flat + x_indices_flat * kernel_size * kernel_size + ((-4) * kernel_size * x_indices_flat)),
            r_mask & x_mask,
            eviction_policy='evict_first',
            other=0.0
        )
        mean_values = tl.load(
            mean_ptr + (r_indices_flat + 4 * x_indices_flat + x_indices_flat * kernel_size * kernel_size + ((-4) * kernel_size * x_indices_flat)),
            r_mask & x_mask,
            eviction_policy='evict_first',
            other=0.0
        )
        scale_factor = 0.5
        scaled_grad_input = grad_input_values * scale_factor
        broadcast_scaled_grad_input = tl.broadcast_to(scaled_grad_input, [XBLOCK, RBLOCK])
        temp_sum_mean += broadcast_scaled_grad_input
        temp_sum_mean = tl.where(r_mask & x_mask, temp_sum_mean, temp_sum_mean)

        grad_variance = grad_input_values - variance_values
        scaled_grad_variance = scaled_grad_input * grad_variance
        broadcast_scaled_grad_variance = tl.broadcast_to(scaled_grad_variance, [XBLOCK, RBLOCK])
        temp_sum_variance += broadcast_scaled_grad_variance
        temp_sum_variance = tl.where(r_mask & x_mask, temp_sum_variance, temp_sum_variance)

    sum_mean = tl.sum(temp_sum_mean, 1)[:, None]
    sum_variance = tl.sum(temp_sum_variance, 1)[:, None]
    tl.store(output_grad_mean_ptr + (x_indices_flat), sum_mean, x_mask)
    tl.store(output_grad_variance_ptr + (x_indices_flat), sum_variance, x_mask)