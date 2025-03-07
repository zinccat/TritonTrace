# From: 20_MobileNetV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_27(
    input_ptr_mean, input_ptr_var, input_ptr_count, input_ptr_running_mean, input_ptr_running_var,
    output_ptr_normalized, output_ptr_running_mean, output_ptr_running_var, output_ptr_mean,
    xnumel, rnumel, XBLOCK: tl.constexpr
):
    xnumel = 144
    rnumel = 62
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex

    mean = tl.load(input_ptr_mean + (x0 + 144 * r1), rmask & xmask, other=0.0)
    variance = tl.load(input_ptr_var + (x0 + 144 * r1), rmask & xmask, other=0.0)
    count = tl.load(input_ptr_count + (x0 + 144 * r1), rmask & xmask, other=0.0)
    running_mean = tl.load(input_ptr_running_mean + (x0), xmask, eviction_policy='evict_last')
    running_var = tl.load(input_ptr_running_var + (x0), xmask, eviction_policy='evict_last')

    mean_broadcast = tl.broadcast_to(mean, [XBLOCK, RBLOCK])
    variance_broadcast = tl.broadcast_to(variance, [XBLOCK, RBLOCK])
    count_broadcast = tl.broadcast_to(count, [XBLOCK, RBLOCK])

    mean_masked = tl.where(rmask & xmask, mean_broadcast, 0)
    variance_masked = tl.where(rmask & xmask, variance_broadcast, 0)
    count_masked = tl.where(rmask & xmask, count_broadcast, 0)

    mean_accum, variance_accum, count_accum = triton_helpers.welford(
        mean_masked, variance_masked, count_masked, 1
    )

    mean_accum_broadcast = mean_accum[:, None]
    variance_accum_broadcast = variance_accum[:, None]

    variance_normalized = 7840.0
    epsilon = 1e-05
    variance_adjusted = variance_accum_broadcast / variance_normalized
    variance_adjusted += epsilon
    inv_stddev = tl.extra.cuda.libdevice.rsqrt(variance_adjusted)

    momentum = 0.1
    mean_scaled = mean_accum_broadcast * momentum

    running_mean_momentum = 0.9
    updated_running_mean = running_mean * running_mean_momentum
    updated_running_mean += mean_scaled

    variance_scale = 1.0001275672917465
    variance_scaled = variance_adjusted * variance_scale
    variance_scaled *= momentum

    updated_running_var = running_var * running_mean_momentum
    updated_running_var += variance_scaled

    tl.store(output_ptr_normalized + (x0), inv_stddev, xmask)
    tl.store(output_ptr_running_mean + (x0), updated_running_mean, xmask)
    tl.store(output_ptr_running_var + (x0), updated_running_var, xmask)
    tl.store(output_ptr_mean + (x0), mean_accum_broadcast, xmask)