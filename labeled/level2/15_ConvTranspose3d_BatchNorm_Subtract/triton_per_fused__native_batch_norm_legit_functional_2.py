# From: 15_ConvTranspose3d_BatchNorm_Subtract

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_2(
    input_ptr_mean, input_ptr_var, input_ptr_count, input_ptr_running_mean, input_ptr_running_var,
    output_ptr_normalized, output_ptr_running_mean, output_ptr_running_var, output_ptr_mean, output_ptr_var, 
    xnumel, rnumel, XBLOCK: tl.constexpr
):
    xnumel = 32
    rnumel = 11
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex

    # Load input data
    input_mean = tl.load(input_ptr_mean + (x0 + (32 * r1)), rmask & xmask, other=0.0)
    input_var = tl.load(input_ptr_var + (x0 + (32 * r1)), rmask & xmask, other=0.0)
    input_count = tl.load(input_ptr_count + (x0 + (32 * r1)), rmask & xmask, other=0.0)
    running_mean = tl.load(input_ptr_running_mean + (x0), xmask, eviction_policy='evict_last')
    running_var = tl.load(input_ptr_running_var + (x0), xmask, eviction_policy='evict_last')

    # Broadcast inputs
    broadcast_mean = tl.broadcast_to(input_mean, [XBLOCK, RBLOCK])
    broadcast_var = tl.broadcast_to(input_var, [XBLOCK, RBLOCK])
    broadcast_count = tl.broadcast_to(input_count, [XBLOCK, RBLOCK])

    # Apply mask
    masked_mean = tl.where(rmask & xmask, broadcast_mean, 0)
    masked_var = tl.where(rmask & xmask, broadcast_var, 0)
    masked_count = tl.where(rmask & xmask, broadcast_count, 0)

    # Compute Welford's algorithm
    mean, var, count = triton_helpers.welford(masked_mean, masked_var, masked_count, 1)

    # Normalize variance
    mean_reshaped = mean[:, None]
    var_reshaped = var[:, None]
    normalized_var = 1968624.0
    epsilon = 1e-05
    adjusted_var = var_reshaped / normalized_var + epsilon
    inv_std = tl.extra.cuda.libdevice.rsqrt(adjusted_var)

    # Update running statistics
    momentum = 0.1
    decay = 0.9
    updated_running_mean = running_mean * decay + mean_reshaped * momentum
    updated_running_var = running_var * decay + var_reshaped * momentum

    # Store results
    tl.store(output_ptr_normalized + (x0), inv_std, xmask)
    tl.store(output_ptr_running_mean + (x0), updated_running_mean, xmask)
    tl.store(output_ptr_running_var + (x0), updated_running_var, xmask)
    tl.store(output_ptr_mean + (x0), mean_reshaped, xmask)
    tl.store(output_ptr_var + (x0), var_reshaped, xmask)