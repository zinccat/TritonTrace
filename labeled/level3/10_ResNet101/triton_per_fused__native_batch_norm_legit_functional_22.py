# From: 10_ResNet101

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_22(
    input_mean_ptr, input_var_ptr, input_x_ptr, running_mean_ptr, running_var_ptr,
    output_mean_ptr, output_var_ptr, output_x_ptr, output_running_mean_ptr, output_running_var_ptr,
    xnumel, rnumel, XBLOCK: tl.constexpr
):
    xnumel = 128
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex

    input_x = tl.load(input_x_ptr + (x0 + 128 * r1), xmask, other=0.0)
    input_mean = tl.load(input_mean_ptr + (x0 + 128 * r1), xmask, other=0.0)
    input_var = tl.load(input_var_ptr + (x0 + 128 * r1), xmask, other=0.0)
    running_mean = tl.load(running_mean_ptr + (x0), xmask, eviction_policy='evict_last')
    running_var = tl.load(running_var_ptr + (x0), xmask, eviction_policy='evict_last')

    broadcast_input_x = tl.broadcast_to(input_x, [XBLOCK, RBLOCK])
    broadcast_input_mean = tl.broadcast_to(input_mean, [XBLOCK, RBLOCK])
    broadcast_input_var = tl.broadcast_to(input_var, [XBLOCK, RBLOCK])

    masked_input_x = tl.where(xmask, broadcast_input_x, 0)
    masked_input_mean = tl.where(xmask, broadcast_input_mean, 0)
    masked_input_var = tl.where(xmask, broadcast_input_var, 0)

    mean, mean_square, count = triton_helpers.welford(masked_input_x, masked_input_mean, masked_input_var, 1)
    mean = mean[:, None]
    mean_square = mean_square[:, None]

    epsilon = 1e-05
    variance = mean_square / 31360.0
    variance += epsilon
    inv_std = tl.extra.cuda.libdevice.rsqrt(variance)

    momentum = 0.1
    updated_mean = mean * momentum

    running_mean_momentum = 0.9
    updated_running_mean = running_mean * running_mean_momentum
    updated_running_mean += updated_mean

    bias_correction = 1.0000318887719635
    corrected_mean = mean_square * bias_correction
    corrected_mean *= momentum

    updated_running_var = running_var * running_mean_momentum
    updated_running_var += corrected_mean

    tl.store(output_var_ptr + (x0), inv_std, xmask)
    tl.store(output_running_mean_ptr + (x0), updated_running_mean, xmask)
    tl.store(output_running_var_ptr + (x0), updated_running_var, xmask)
    tl.store(output_mean_ptr + (x0), mean, xmask)
    tl.store(output_x_ptr + (x0), mean_square, xmask)