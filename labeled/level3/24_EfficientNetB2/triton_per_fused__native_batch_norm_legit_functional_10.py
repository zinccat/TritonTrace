# From: 24_EfficientNetB2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_10(
    input_mean_ptr, input_var_ptr, input_x_ptr, running_mean_ptr, running_var_ptr,
    output_mean_ptr, output_var_ptr, output_x_ptr, output_xn_ptr, xnumel, rnumel, XBLOCK: tl.constexpr
):
    xnumel = 96
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex

    input_mean = tl.load(input_mean_ptr + (x0 + 96 * r1), xmask, other=0.0)
    input_var = tl.load(input_var_ptr + (x0 + 96 * r1), xmask, other=0.0)
    input_x = tl.load(input_x_ptr + (x0 + 96 * r1), xmask, other=0.0)
    running_mean = tl.load(running_mean_ptr + (x0), xmask, eviction_policy='evict_last')
    running_var = tl.load(running_var_ptr + (x0), xmask, eviction_policy='evict_last')

    input_mean_broadcast = tl.broadcast_to(input_mean, [XBLOCK, RBLOCK])
    input_var_broadcast = tl.broadcast_to(input_var, [XBLOCK, RBLOCK])
    input_x_broadcast = tl.broadcast_to(input_x, [XBLOCK, RBLOCK])

    masked_input_mean = tl.where(xmask, input_mean_broadcast, 0)
    masked_input_var = tl.where(xmask, input_var_broadcast, 0)
    masked_input_x = tl.where(xmask, input_x_broadcast, 0)

    mean, var, _ = triton_helpers.welford(masked_input_mean, masked_input_var, masked_input_x, 1)

    mean_reshaped = mean[:, None]
    var_reshaped = var[:, None]

    normalized_var = 25088.0
    epsilon = 1e-05
    adjusted_var = var_reshaped / normalized_var
    adjusted_var_with_epsilon = adjusted_var + epsilon

    inv_std = tl.extra.cuda.libdevice.rsqrt(adjusted_var_with_epsilon)

    momentum = 0.1
    scaled_mean = mean_reshaped * momentum

    running_mean_momentum = 0.9
    updated_running_mean = running_mean * running_mean_momentum
    new_running_mean = scaled_mean + updated_running_mean

    variance_scale = 1.0000398612827361
    scaled_variance = adjusted_var * variance_scale
    scaled_variance_with_momentum = scaled_variance * momentum

    updated_running_var = running_var * running_mean_momentum
    new_running_var = scaled_variance_with_momentum + updated_running_var

    tl.store(output_var_ptr + (x0), inv_std, xmask)
    tl.store(output_mean_ptr + (x0), new_running_mean, xmask)
    tl.store(output_xn_ptr + (x0), new_running_var, xmask)
    tl.store(output_x_ptr + (x0), mean_reshaped, xmask)