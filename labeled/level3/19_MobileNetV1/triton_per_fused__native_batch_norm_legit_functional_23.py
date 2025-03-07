# From: 19_MobileNetV1

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_23(
    input_mean_ptr, input_var_ptr, input_x_ptr, running_mean_ptr, running_var_ptr,
    output_mean_ptr, output_var_ptr, output_x_ptr, output_running_mean_ptr, output_running_var_ptr,
    xnumel, rnumel, XBLOCK: tl.constexpr
):
    xnumel = 256
    rnumel = 62
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex

    input_mean = tl.load(input_mean_ptr + (x0 + 256 * r1), rmask & xmask, other=0.0)
    input_var = tl.load(input_var_ptr + (x0 + 256 * r1), rmask & xmask, other=0.0)
    input_x = tl.load(input_x_ptr + (x0 + 256 * r1), rmask & xmask, other=0.0)
    running_mean = tl.load(running_mean_ptr + (x0), xmask, eviction_policy='evict_last')
    running_var = tl.load(running_var_ptr + (x0), xmask, eviction_policy='evict_last')

    broadcast_mean = tl.broadcast_to(input_mean, [XBLOCK, RBLOCK])
    broadcast_var = tl.broadcast_to(input_var, [XBLOCK, RBLOCK])
    broadcast_x = tl.broadcast_to(input_x, [XBLOCK, RBLOCK])

    masked_mean = tl.where(rmask & xmask, broadcast_mean, 0)
    masked_var = tl.where(rmask & xmask, broadcast_var, 0)
    masked_x = tl.where(rmask & xmask, broadcast_x, 0)

    mean, var, _ = triton_helpers.welford(masked_mean, masked_var, masked_x, 1)

    mean_reshaped = mean[:, None]
    var_reshaped = var[:, None]

    epsilon = 1e-05
    normalized_var = var_reshaped / 7840.0
    adjusted_var = normalized_var + epsilon
    inv_std = tl.extra.cuda.libdevice.rsqrt(adjusted_var)

    momentum = 0.1
    mean_scaled = mean_reshaped * momentum

    running_mean_momentum = 0.9
    updated_running_mean = mean_scaled + running_mean * running_mean_momentum

    variance_scale = 1.0001275672917465
    normalized_variance = normalized_var * variance_scale
    variance_scaled = normalized_variance * momentum

    updated_running_var = variance_scaled + running_var * running_mean_momentum

    tl.store(output_var_ptr + (x0), inv_std, xmask)
    tl.store(output_running_mean_ptr + (x0), updated_running_mean, xmask)
    tl.store(output_running_var_ptr + (x0), updated_running_var, xmask)
    tl.store(output_mean_ptr + (x0), mean_reshaped, xmask)
    tl.store(output_x_ptr + (x0), var_reshaped, xmask)