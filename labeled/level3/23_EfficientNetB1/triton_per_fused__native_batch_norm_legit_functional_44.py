# From: 23_EfficientNetB1

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_44(
    input_mean_ptr, input_var_ptr, input_count_ptr, running_mean_ptr, running_var_ptr,
    output_mean_ptr, output_var_ptr, output_running_mean_ptr, output_running_var_ptr,
    xnumel, rnumel, XBLOCK: tl.constexpr
):
    xnumel = 480
    rnumel = 18
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex

    input_mean = tl.load(input_mean_ptr + (x0 + 480 * r1), rmask & xmask, other=0.0)
    input_var = tl.load(input_var_ptr + (x0 + 480 * r1), rmask & xmask, other=0.0)
    input_count = tl.load(input_count_ptr + (x0 + 480 * r1), rmask & xmask, other=0.0)
    running_mean = tl.load(running_mean_ptr + (x0), xmask, eviction_policy='evict_last')
    running_var = tl.load(running_var_ptr + (x0), xmask, eviction_policy='evict_last')

    broadcast_mean = tl.broadcast_to(input_mean, [XBLOCK, RBLOCK])
    broadcast_var = tl.broadcast_to(input_var, [XBLOCK, RBLOCK])
    broadcast_count = tl.broadcast_to(input_count, [XBLOCK, RBLOCK])

    masked_mean = tl.where(rmask & xmask, broadcast_mean, 0)
    masked_var = tl.where(rmask & xmask, broadcast_var, 0)
    masked_count = tl.where(rmask & xmask, broadcast_count, 0)

    mean, var, count = triton_helpers.welford(masked_mean, masked_var, masked_count, 1)

    reshaped_mean = mean[:, None]
    reshaped_var = var[:, None]

    epsilon = 1e-05
    normalized_var = reshaped_var / 2250.0
    adjusted_var = normalized_var + epsilon
    inv_std = tl.extra.cuda.libdevice.rsqrt(adjusted_var)

    momentum = 0.1
    scaled_mean = reshaped_mean * momentum

    running_mean_momentum = 0.9
    updated_running_mean = running_mean * running_mean_momentum + scaled_mean

    var_scale_factor = 1.0004446420631392
    scaled_var = normalized_var * var_scale_factor
    scaled_var_momentum = scaled_var * momentum

    updated_running_var = running_var * running_mean_momentum + scaled_var_momentum

    tl.store(output_var_ptr + (x0), inv_std, xmask)
    tl.store(output_running_mean_ptr + (x0), updated_running_mean, xmask)
    tl.store(output_running_var_ptr + (x0), updated_running_var, xmask)
    tl.store(output_mean_ptr + (x0), reshaped_mean, xmask)