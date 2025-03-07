# From: 35_GroupNorm_

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_native_group_norm_1per_fused_native_group_norm_1(input_ptr_mean, input_ptr_var, input_ptr_input, output_ptr_mean, output_ptr_var, output_ptr_normalized, kernel_size, num_elements, num_channels, XBLOCK: tl.constexpr):
    num_channels = 3
    RBLOCK: tl.constexpr = 4
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements
    r_indices = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_indices < num_channels
    channel_indices = r_indices
    element_indices = x_indices

    mean_channel_0 = tl.load(input_ptr_mean + (channel_indices + 3 * element_indices), r_mask & x_mask, other=0.0)
    mean_channel_1 = tl.load(input_ptr_var + (channel_indices + 3 * element_indices), r_mask & x_mask, other=0.0)
    mean_channel_2 = tl.load(input_ptr_input + (channel_indices + 3 * element_indices), r_mask & x_mask, other=0.0)

    broadcast_mean_0 = tl.broadcast_to(mean_channel_0, [XBLOCK, RBLOCK])
    broadcast_mean_1 = tl.broadcast_to(mean_channel_1, [XBLOCK, RBLOCK])
    broadcast_mean_2 = tl.broadcast_to(mean_channel_2, [XBLOCK, RBLOCK])

    masked_mean_0 = tl.where(r_mask & x_mask, broadcast_mean_0, 0)
    masked_mean_1 = tl.where(r_mask & x_mask, broadcast_mean_1, 0)
    masked_mean_2 = tl.where(r_mask & x_mask, broadcast_mean_2, 0)

    mean, variance, _ = triton_helpers.welford(masked_mean_0, masked_mean_1, masked_mean_2, 1)

    reshaped_mean = mean[:, None]
    reshaped_variance = variance[:, None]

    normalization_factor = 8 * kernel_size * kernel_size
    normalization_factor_float = normalization_factor.to(tl.float32)
    normalized_variance = reshaped_variance / normalization_factor_float

    epsilon = 1e-05
    adjusted_variance = normalized_variance + epsilon
    reciprocal_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(adjusted_variance)

    tl.store(output_ptr_normalized + (element_indices), reciprocal_sqrt_variance, x_mask)
    tl.store(output_ptr_mean + (element_indices), reshaped_mean, x_mask)
    tl.store(output_ptr_var + (element_indices), reshaped_variance, x_mask)