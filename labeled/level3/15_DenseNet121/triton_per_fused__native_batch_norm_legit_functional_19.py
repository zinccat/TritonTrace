# From: 15_DenseNet121

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_19(
    input_mean_ptr, input_var_ptr, input_x_ptr, running_mean_ptr, running_var_ptr,
    output_mean_ptr, output_var_ptr, output_running_mean_ptr, output_running_var_ptr,
    xnumel, rnumel, XBLOCK: tl.constexpr
):
    xnumel = 160
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex

    input_mean = tl.load(input_mean_ptr + (x0 + 160 * r1), xmask, other=0.0)
    input_var = tl.load(input_var_ptr + (x0 + 160 * r1), xmask, other=0.0)
    input_x = tl.load(input_x_ptr + (x0 + 160 * r1), xmask, other=0.0)
    running_mean = tl.load(running_mean_ptr + (x0), xmask, eviction_policy='evict_last')
    running_var = tl.load(running_var_ptr + (x0), xmask, eviction_policy='evict_last')

    broadcast_mean = tl.broadcast_to(input_mean, [XBLOCK, RBLOCK])
    broadcast_var = tl.broadcast_to(input_var, [XBLOCK, RBLOCK])
    broadcast_x = tl.broadcast_to(input_x, [XBLOCK, RBLOCK])

    masked_mean = tl.where(xmask, broadcast_mean, 0)
    masked_var = tl.where(xmask, broadcast_var, 0)
    masked_x = tl.where(xmask, broadcast_x, 0)

    mean_update, var_update, count_update = triton_helpers.welford(masked_mean, masked_var, masked_x, 1)

    reshaped_mean_update = mean_update[:, None]
    reshaped_var_update = var_update[:, None]

    rsqrt_denominator = 31360.0
    epsilon = 1e-05
    normalized_var = reshaped_var_update / rsqrt_denominator
    adjusted_var = normalized_var + epsilon
    inv_std = tl.extra.cuda.libdevice.rsqrt(adjusted_var)

    momentum = 0.1
    mean_momentum = reshaped_mean_update * momentum

    running_mean_momentum = 0.9
    updated_running_mean = running_mean * running_mean_momentum
    new_running_mean = mean_momentum + updated_running_mean

    variance_scale = 1.0000318887719635
    scaled_var = normalized_var * variance_scale
    var_momentum = scaled_var * momentum

    updated_running_var = running_var * running_mean_momentum
    new_running_var = var_momentum + updated_running_var

    tl.store(output_var_ptr + (x0), inv_std, xmask)
    tl.store(output_running_mean_ptr + (x0), new_running_mean, xmask)
    tl.store(output_running_var_ptr + (x0), new_running_var, xmask)
    tl.store(output_mean_ptr + (x0), reshaped_mean_update, xmask)
    tl.store(output_var_ptr + (x0), reshaped_var_update, xmask)