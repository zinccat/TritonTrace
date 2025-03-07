# From: 46_NetVladWithGhostClusters

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_2(
    input_ptr_mean, input_ptr_var, input_ptr_count, input_ptr_running_mean, input_ptr_running_var,
    output_ptr_normalized, output_ptr_running_mean, output_ptr_running_var, output_ptr_mean,
    xnumel, rnumel, XBLOCK: tl.constexpr
):
    xnumel = 48
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
    input_mean = tl.load(input_ptr_mean + (x0 + 48 * r1), rmask & xmask, other=0.0)
    input_var = tl.load(input_ptr_var + (x0 + 48 * r1), rmask & xmask, other=0.0)
    input_count = tl.load(input_ptr_count + (x0 + 48 * r1), rmask & xmask, other=0.0)
    running_mean = tl.load(input_ptr_running_mean + (x0), xmask, eviction_policy='evict_last')
    running_var = tl.load(input_ptr_running_var + (x0), xmask, eviction_policy='evict_last')

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

    # Compute normalization parameters
    epsilon = 1e-05
    inv_std = tl.extra.cuda.libdevice.rsqrt(reshaped_var / 3200.0 + epsilon)
    momentum = 0.1
    adjusted_var = (reshaped_var / 3200.0) * 1.000312597686777 * momentum
    running_mean_factor = momentum
    running_var_factor = 0.9

    # Update running statistics
    updated_running_mean = running_mean * running_mean_factor + reshaped_mean * running_mean_factor
    updated_running_var = running_var * running_var_factor + adjusted_var + reshaped_var * running_mean_factor

    # Store results
    tl.store(output_ptr_normalized + (x0), inv_std, xmask)
    tl.store(output_ptr_running_mean + (x0), updated_running_mean, xmask)
    tl.store(output_ptr_running_var + (x0), updated_running_var, xmask)
    tl.store(output_ptr_mean + (x0), reshaped_mean, xmask)