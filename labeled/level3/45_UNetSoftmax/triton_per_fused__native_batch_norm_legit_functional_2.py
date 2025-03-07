# From: 45_UNetSoftmax

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_2(
    input_mean_ptr, input_var_ptr, input_x_ptr, running_mean_ptr, running_var_ptr,
    output_mean_ptr, output_var_ptr, output_x_ptr, output_running_mean_ptr, output_running_var_ptr,
    xnumel, rnumel, XBLOCK: tl.constexpr
):
    xnumel = 64
    rnumel = 6
    RBLOCK: tl.constexpr = 8
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_index = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_index < rnumel
    r1 = r_index
    x0 = x_index

    input_mean = tl.load(input_mean_ptr + (x0 + 64 * r1), r_mask & x_mask, other=0.0)
    input_var = tl.load(input_var_ptr + (x0 + 64 * r1), r_mask & x_mask, other=0.0)
    input_x = tl.load(input_x_ptr + (x0 + 64 * r1), r_mask & x_mask, other=0.0)
    running_mean = tl.load(running_mean_ptr + (x0), x_mask, eviction_policy='evict_last')
    running_var = tl.load(running_var_ptr + (x0), x_mask, eviction_policy='evict_last')

    broadcast_mean = tl.broadcast_to(input_mean, [XBLOCK, RBLOCK])
    broadcast_var = tl.broadcast_to(input_var, [XBLOCK, RBLOCK])
    broadcast_x = tl.broadcast_to(input_x, [XBLOCK, RBLOCK])

    masked_mean = tl.where(r_mask & x_mask, broadcast_mean, 0)
    masked_var = tl.where(r_mask & x_mask, broadcast_var, 0)
    masked_x = tl.where(r_mask & x_mask, broadcast_x, 0)

    mean, var, _ = triton_helpers.welford(masked_mean, masked_var, masked_x, 1)

    mean_reshaped = mean[:, None]
    var_reshaped = var[:, None]

    normalization_factor = 262144.0
    epsilon = 1e-05
    adjusted_var = var_reshaped / normalization_factor
    adjusted_var += epsilon
    inv_std = tl.extra.cuda.libdevice.rsqrt(adjusted_var)

    momentum = 0.1
    running_mean_update = mean_reshaped * momentum
    running_mean_scaled = running_mean * 0.9
    updated_running_mean = running_mean_update + running_mean_scaled

    variance_scale = 1.0000038147118175
    scaled_variance = adjusted_var * variance_scale
    scaled_variance_update = scaled_variance * momentum
    running_var_scaled = running_var * 0.9
    updated_running_var = scaled_variance_update + running_var_scaled

    tl.store(output_var_ptr + (x0), inv_std, x_mask)
    tl.store(output_running_mean_ptr + (x0), updated_running_mean, x_mask)
    tl.store(output_running_var_ptr + (x0), updated_running_var, x_mask)
    tl.store(output_mean_ptr + (x0), mean_reshaped, x_mask)
    tl.store(output_var_ptr + (x0), var_reshaped, x_mask)