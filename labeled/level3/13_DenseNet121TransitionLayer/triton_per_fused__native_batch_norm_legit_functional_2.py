# From: 13_DenseNet121TransitionLayer

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_2(
    input_mean_ptr, input_var_ptr, input_count_ptr, input_running_mean_ptr, 
    input_running_var_ptr, output_mean_ptr, output_var_ptr, output_running_mean_ptr, 
    output_running_var_ptr, xnumel, rnumel, XBLOCK: tl.constexpr
):
    xnumel = 32
    rnumel = 10
    RBLOCK: tl.constexpr = 16
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_index = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_index < rnumel
    r1 = r_index
    x0 = x_index

    # Load input data
    input_mean = tl.load(input_mean_ptr + (x0 + 32 * r1), r_mask & x_mask, other=0.0)
    input_var = tl.load(input_var_ptr + (x0 + 32 * r1), r_mask & x_mask, other=0.0)
    input_count = tl.load(input_count_ptr + (x0 + 32 * r1), r_mask & x_mask, other=0.0)
    input_running_mean = tl.load(input_running_mean_ptr + (x0), x_mask, eviction_policy='evict_last')
    input_running_var = tl.load(input_running_var_ptr + (x0), x_mask, eviction_policy='evict_last')

    # Broadcast loaded data
    broadcast_mean = tl.broadcast_to(input_mean, [XBLOCK, RBLOCK])
    broadcast_var = tl.broadcast_to(input_var, [XBLOCK, RBLOCK])
    broadcast_count = tl.broadcast_to(input_count, [XBLOCK, RBLOCK])

    # Apply masks
    masked_mean = tl.where(r_mask & x_mask, broadcast_mean, 0)
    masked_var = tl.where(r_mask & x_mask, broadcast_var, 0)
    masked_count = tl.where(r_mask & x_mask, broadcast_count, 0)

    # Compute Welford's algorithm
    mean, var, count = triton_helpers.welford(masked_mean, masked_var, masked_count, 1)

    # Reshape mean and variance
    reshaped_mean = mean[:, None]
    reshaped_var = var[:, None]

    # Compute running variance
    epsilon = 1e-05
    inv_std = tl.extra.cuda.libdevice.rsqrt(reshaped_var / 501760.0 + epsilon)

    # Update running mean and variance
    momentum = 0.1
    decay = 0.9
    updated_running_mean = input_running_mean * decay + reshaped_mean * momentum
    updated_running_var = input_running_var * decay + (reshaped_var * momentum / 501760.0 * 1.0000019929886659)

    # Store results
    tl.store(output_mean_ptr + (x0), inv_std, x_mask)
    tl.store(output_var_ptr + (x0), updated_running_var, x_mask)
    tl.store(output_running_mean_ptr + (x0), updated_running_mean, x_mask)
    tl.store(output_running_var_ptr + (x0), reshaped_var, x_mask)
    tl.store(output_mean_ptr + (x0), reshaped_mean, x_mask)
    tl.store(output_var_ptr + (x0), reshaped_var, x_mask)