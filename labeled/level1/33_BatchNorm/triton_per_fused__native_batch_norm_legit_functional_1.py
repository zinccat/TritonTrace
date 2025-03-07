# From: 33_BatchNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_1(
    input_mean_ptr, input_var_ptr, input_x_ptr, running_mean_ptr, running_var_ptr,
    output_mean_ptr, output_var_ptr, output_x_ptr, output_running_mean_ptr, output_running_var_ptr,
    xnumel, rnumel, XBLOCK: tl.constexpr
):
    xnumel = 64
    rnumel = 6
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex

    # Load input tensors
    input_mean = tl.load(input_mean_ptr + (x0 + (64 * r1)), rmask & xmask, other=0.0)
    input_var = tl.load(input_var_ptr + (x0 + (64 * r1)), rmask & xmask, other=0.0)
    input_x = tl.load(input_x_ptr + (x0 + (64 * r1)), rmask & xmask, other=0.0)
    running_mean = tl.load(running_mean_ptr + (x0), xmask, eviction_policy='evict_last')
    running_var = tl.load(running_var_ptr + (x0), xmask, eviction_policy='evict_last')

    # Broadcast loaded values
    broadcast_mean = tl.broadcast_to(input_mean, [XBLOCK, RBLOCK])
    broadcast_var = tl.broadcast_to(input_var, [XBLOCK, RBLOCK])
    broadcast_x = tl.broadcast_to(input_x, [XBLOCK, RBLOCK])

    # Apply masks
    masked_mean = tl.where(rmask & xmask, broadcast_mean, 0)
    masked_var = tl.where(rmask & xmask, broadcast_var, 0)
    masked_x = tl.where(rmask & xmask, broadcast_x, 0)

    # Compute Welford's algorithm
    mean, var, _ = triton_helpers.welford(masked_mean, masked_var, masked_x, 1)

    # Normalize variance
    normalized_var = var[:, None]
    variance_scale = 1048576.0
    scaled_variance = normalized_var / variance_scale
    epsilon = 1e-05
    adjusted_variance = scaled_variance + epsilon
    inv_std = tl.extra.cuda.libdevice.rsqrt(adjusted_variance)

    # Update running mean and variance
    momentum = 0.1
    decay = 0.9
    updated_running_mean = running_mean * decay + mean[:, None] * momentum
    updated_running_var = running_var * decay + var[:, None] * momentum

    # Store results
    tl.store(output_x_ptr + (x0), inv_std, xmask)
    tl.store(output_running_mean_ptr + (x0), updated_running_mean, xmask)
    tl.store(output_running_var_ptr + (x0), updated_running_var, xmask)
    tl.store(output_mean_ptr + (x0), mean, xmask)
    tl.store(output_var_ptr + (x0), var, xmask)