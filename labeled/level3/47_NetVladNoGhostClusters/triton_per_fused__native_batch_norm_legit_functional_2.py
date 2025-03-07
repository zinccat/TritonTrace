# From: 47_NetVladNoGhostClusters

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_2(
    input_mean_ptr, input_var_ptr, input_count_ptr, running_mean_ptr, running_var_ptr,
    output_mean_ptr, output_var_ptr, output_count_ptr, output_data_ptr,
    xnumel, rnumel, XBLOCK: tl.constexpr
):
    xnumel = 32
    rnumel = 25
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex

    # Load input data with masking
    input_mean = tl.load(input_mean_ptr + (x0 + 32 * r1), rmask & xmask, other=0.0)
    input_var = tl.load(input_var_ptr + (x0 + 32 * r1), rmask & xmask, other=0.0)
    input_count = tl.load(input_count_ptr + (x0 + 32 * r1), rmask & xmask, other=0.0)

    # Load running mean and variance with eviction policy
    running_mean = tl.load(running_mean_ptr + (x0), xmask, eviction_policy='evict_last')
    running_var = tl.load(running_var_ptr + (x0), xmask, eviction_policy='evict_last')

    # Broadcast loaded data
    broadcast_mean = tl.broadcast_to(input_mean, [XBLOCK, RBLOCK])
    broadcast_var = tl.broadcast_to(input_var, [XBLOCK, RBLOCK])
    broadcast_count = tl.broadcast_to(input_count, [XBLOCK, RBLOCK])

    # Apply masks
    masked_mean = tl.where(rmask & xmask, broadcast_mean, 0)
    masked_var = tl.where(rmask & xmask, broadcast_var, 0)
    masked_count = tl.where(rmask & xmask, broadcast_count, 0)

    # Compute Welford's algorithm
    mean, var, count = triton_helpers.welford(masked_mean, masked_var, masked_count, 1)

    # Reshape mean and variance
    reshaped_mean = mean[:, None]
    reshaped_var = var[:, None]

    # Compute inverse square root of variance
    epsilon = 1e-05
    adjusted_var = reshaped_var / 3200.0 + epsilon
    inv_sqrt_var = tl.extra.cuda.libdevice.rsqrt(adjusted_var)

    # Update running mean and variance
    momentum = 0.1
    decay = 0.9
    adjusted_mean = reshaped_mean * momentum
    updated_running_mean = running_mean * decay + adjusted_mean
    updated_running_var = running_var * decay + reshaped_var * momentum

    # Store results
    tl.store(output_mean_ptr + (x0), inv_sqrt_var, xmask)
    tl.store(output_var_ptr + (x0), updated_running_mean, xmask)
    tl.store(output_count_ptr + (x0), updated_running_var, xmask)
    tl.store(output_data_ptr + (x0), reshaped_mean, xmask)