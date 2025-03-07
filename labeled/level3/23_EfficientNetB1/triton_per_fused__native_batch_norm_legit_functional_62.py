# From: 23_EfficientNetB1

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_62(
    input_mean_ptr, input_var_ptr, input_count_ptr, running_mean_ptr, running_var_ptr,
    output_mean_ptr, output_var_ptr, output_running_mean_ptr, output_running_var_ptr,
    xnumel, rnumel, XBLOCK: tl.constexpr
):
    xnumel = 320
    rnumel = 5
    RBLOCK: tl.constexpr = 8
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_index = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_index < rnumel
    r1 = r_index
    x0 = x_index

    input_mean = tl.load(input_mean_ptr + (x0 + 320 * r1), r_mask & x_mask, other=0.0)
    input_var = tl.load(input_var_ptr + (x0 + 320 * r1), r_mask & x_mask, other=0.0)
    input_count = tl.load(input_count_ptr + (x0 + 320 * r1), r_mask & x_mask, other=0.0)
    running_mean = tl.load(running_mean_ptr + (x0), x_mask, eviction_policy='evict_last')
    running_var = tl.load(running_var_ptr + (x0), x_mask, eviction_policy='evict_last')

    broadcast_mean = tl.broadcast_to(input_mean, [XBLOCK, RBLOCK])
    broadcast_var = tl.broadcast_to(input_var, [XBLOCK, RBLOCK])
    broadcast_count = tl.broadcast_to(input_count, [XBLOCK, RBLOCK])

    masked_mean = tl.where(r_mask & x_mask, broadcast_mean, 0)
    masked_var = tl.where(r_mask & x_mask, broadcast_var, 0)
    masked_count = tl.where(r_mask & x_mask, broadcast_count, 0)

    mean, var, count = triton_helpers.welford(masked_mean, masked_var, masked_count, 1)

    reshaped_mean = mean[:, None]
    reshaped_var = var[:, None]

    normalized_var = reshaped_var / 640.0
    epsilon = 1e-05
    adjusted_var = normalized_var + epsilon
    inv_std = tl.extra.cuda.libdevice.rsqrt(adjusted_var)

    momentum = 0.1
    updated_mean = reshaped_mean * momentum

    decay = 0.9
    decayed_running_mean = running_mean * decay
    new_running_mean = updated_mean + decayed_running_mean

    scale_factor = 1.001564945226917
    scaled_var = normalized_var * scale_factor
    scaled_updated_mean = scaled_var * momentum

    decayed_running_var = running_var * decay
    new_running_var = scaled_updated_mean + decayed_running_var

    tl.store(output_var_ptr + (x0), inv_std, x_mask)
    tl.store(output_running_mean_ptr + (x0), new_running_mean, x_mask)
    tl.store(output_running_var_ptr + (x0), new_running_var, x_mask)
    tl.store(output_mean_ptr + (x0), reshaped_mean, x_mask)
    tl.store(output_var_ptr + (x0), reshaped_var, x_mask)