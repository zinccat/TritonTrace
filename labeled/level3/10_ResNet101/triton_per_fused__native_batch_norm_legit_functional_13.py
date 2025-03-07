# From: 10_ResNet101

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_13(
    input_mean_ptr, input_var_ptr, input_x_ptr, running_mean_ptr, running_var_ptr,
    output_mean_ptr, output_var_ptr, output_running_mean_ptr, output_running_var_ptr,
    xnumel, rnumel, XBLOCK: tl.constexpr
):
    xnumel = 64
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex

    input_mean = tl.load(input_mean_ptr + (x0 + 64 * r1), xmask, other=0.0)
    input_var = tl.load(input_var_ptr + (x0 + 64 * r1), xmask, other=0.0)
    input_x = tl.load(input_x_ptr + (x0 + 64 * r1), xmask, other=0.0)
    running_mean = tl.load(running_mean_ptr + (x0), xmask, eviction_policy='evict_last')
    running_var = tl.load(running_var_ptr + (x0), xmask, eviction_policy='evict_last')

    broadcast_mean = tl.broadcast_to(input_mean, [XBLOCK, RBLOCK])
    broadcast_var = tl.broadcast_to(input_var, [XBLOCK, RBLOCK])
    broadcast_x = tl.broadcast_to(input_x, [XBLOCK, RBLOCK])

    masked_mean = tl.where(xmask, broadcast_mean, 0)
    masked_var = tl.where(xmask, broadcast_var, 0)
    masked_x = tl.where(xmask, broadcast_x, 0)

    mean, var, _ = triton_helpers.welford(masked_mean, masked_var, masked_x, 1)

    mean_reshaped = mean[:, None]
    var_reshaped = var[:, None]

    epsilon = 1e-05
    normalized_var = var_reshaped / 31360.0
    adjusted_var = normalized_var + epsilon
    inv_std = tl.extra.cuda.libdevice.rsqrt(adjusted_var)

    momentum = 0.1
    updated_mean = mean * momentum

    decay = 0.9
    decayed_running_mean = running_mean * decay
    new_running_mean = updated_mean + decayed_running_mean

    scale_factor = 1.0000318887719635
    scaled_mean = var_reshaped * scale_factor
    scaled_mean_momentum = scaled_mean * momentum

    decayed_running_var = running_var * decay
    new_running_var = scaled_mean_momentum + decayed_running_var

    tl.store(output_var_ptr + (x0), inv_std, xmask)
    tl.store(output_running_mean_ptr + (x0), new_running_mean, xmask)
    tl.store(output_running_var_ptr + (x0), new_running_var, xmask)
    tl.store(output_mean_ptr + (x0), mean, xmask)
    tl.store(output_var_ptr + (x0), var, xmask)