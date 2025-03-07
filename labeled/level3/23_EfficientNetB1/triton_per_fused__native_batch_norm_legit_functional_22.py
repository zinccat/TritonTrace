# From: 23_EfficientNetB1

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
    xnumel = 24
    rnumel = 3
    RBLOCK: tl.constexpr = 4
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_index = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_index < rnumel
    r1 = r_index
    x0 = x_index

    # Load input tensors
    input_mean = tl.load(input_mean_ptr + (x0 + 24 * r1), r_mask & x_mask, other=0.0)
    input_var = tl.load(input_var_ptr + (x0 + 24 * r1), r_mask & x_mask, other=0.0)
    input_x = tl.load(input_x_ptr + (x0 + 24 * r1), r_mask & x_mask, other=0.0)
    running_mean = tl.load(running_mean_ptr + (x0), x_mask, eviction_policy='evict_last')
    running_var = tl.load(running_var_ptr + (x0), x_mask, eviction_policy='evict_last')

    # Broadcast loaded tensors
    broadcast_mean = tl.broadcast_to(input_mean, [XBLOCK, RBLOCK])
    broadcast_var = tl.broadcast_to(input_var, [XBLOCK, RBLOCK])
    broadcast_x = tl.broadcast_to(input_x, [XBLOCK, RBLOCK])

    # Apply masks
    masked_mean = tl.where(r_mask & x_mask, broadcast_mean, 0)
    masked_var = tl.where(r_mask & x_mask, broadcast_var, 0)
    masked_x = tl.where(r_mask & x_mask, broadcast_x, 0)

    # Compute Welford's algorithm
    mean, var, _ = triton_helpers.welford(masked_mean, masked_var, masked_x, 1)

    # Reshape mean and variance
    reshaped_mean = mean[:, None]
    reshaped_var = var[:, None]

    # Compute inverse square root of variance
    epsilon = 1e-05
    normalized_var = reshaped_var / 36000.0
    adjusted_var = normalized_var + epsilon
    inv_sqrt_var = tl.extra.cuda.libdevice.rsqrt(adjusted_var)

    # Update running mean and variance
    momentum = 0.1
    updated_mean = reshaped_mean * momentum
    decay = 0.9
    running_mean_scaled = running_mean * decay
    new_running_mean = updated_mean + running_mean_scaled

    bias_correction = 1.000027778549404
    normalized_mean = reshaped_var * bias_correction
    updated_var = normalized_mean * momentum
    running_var_scaled = running_var * decay
    new_running_var = updated_var + running_var_scaled

    # Store results
    tl.store(output_var_ptr + (x0), inv_sqrt_var, x_mask)
    tl.store(output_running_mean_ptr + (x0), new_running_mean, x_mask)
    tl.store(output_running_var_ptr + (x0), new_running_var, x_mask)
    tl.store(output_mean_ptr + (x0), reshaped_mean, x_mask)
    tl.store(output_x_ptr + (x0), reshaped_var, x_mask)