# From: 27_RegNet

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_17(
    input_mean_ptr, input_var_ptr, input_count_ptr, input_running_mean_ptr, input_running_var_ptr,
    output_mean_ptr, output_var_ptr, output_running_mean_ptr, output_running_var_ptr, output_count_ptr,
    xnumel, rnumel, XBLOCK: tl.constexpr
):
    xnumel = 128
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex

    # Load input data with masking
    input_mean = tl.load(input_mean_ptr + (x0 + 128 * r1), rmask & xmask, other=0.0)
    input_var = tl.load(input_var_ptr + (x0 + 128 * r1), rmask & xmask, other=0.0)
    input_count = tl.load(input_count_ptr + (x0 + 128 * r1), rmask & xmask, other=0.0)
    input_running_mean = tl.load(input_running_mean_ptr + (x0), xmask, eviction_policy='evict_last')
    input_running_var = tl.load(input_running_var_ptr + (x0), xmask, eviction_policy='evict_last')

    # Broadcast loaded data
    broadcast_mean = tl.broadcast_to(input_mean, [XBLOCK, RBLOCK])
    broadcast_var = tl.broadcast_to(input_var, [XBLOCK, RBLOCK])
    broadcast_count = tl.broadcast_to(input_count, [XBLOCK, RBLOCK])

    # Apply mask
    masked_mean = tl.where(rmask & xmask, broadcast_mean, 0)
    masked_var = tl.where(rmask & xmask, broadcast_var, 0)
    masked_count = tl.where(rmask & xmask, broadcast_count, 0)

    # Compute Welford's algorithm
    mean, var, count = triton_helpers.welford(masked_mean, masked_var, masked_count, 1)

    # Reshape mean and variance
    reshaped_mean = mean[:, None]
    reshaped_var = var[:, None]

    # Compute running variance
    epsilon = 1e-05
    inv_std = tl.extra.cuda.libdevice.rsqrt(reshaped_var / 100352.0 + epsilon)

    # Update running mean and variance
    momentum = 0.1
    running_mean_momentum = 0.9
    running_var_momentum = 0.9

    updated_running_mean = input_running_mean * running_mean_momentum + reshaped_mean * momentum
    updated_running_var = input_running_var * running_var_momentum + (reshaped_var * momentum / 100352.0 * 1.00000996502277)

    # Store results
    tl.store(output_mean_ptr + (x0), reshaped_mean, xmask)
    tl.store(output_var_ptr + (x0), reshaped_var, xmask)
    tl.store(output_running_mean_ptr + (x0), updated_running_mean, xmask)
    tl.store(output_running_var_ptr + (x0), updated_running_var, xmask)
    tl.store(output_count_ptr + (x0), inv_std, xmask)