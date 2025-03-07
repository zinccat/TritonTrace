# From: 23_EfficientNetB1

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_26(
    input_ptr_mean, input_ptr_var, input_ptr_count, input_ptr_running_mean, input_ptr_running_var,
    output_ptr_inv_std, output_ptr_running_mean, output_ptr_running_var, output_ptr_mean,
    xnumel, rnumel, XBLOCK: tl.constexpr
):
    xnumel = 144
    rnumel = 3
    RBLOCK: tl.constexpr = 4
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_index = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_index < rnumel
    r1 = r_index
    x0 = x_index

    # Load input data with masking
    input_mean = tl.load(input_ptr_mean + (x0 + 144 * r1), r_mask & x_mask, other=0.0)
    input_var = tl.load(input_ptr_var + (x0 + 144 * r1), r_mask & x_mask, other=0.0)
    input_count = tl.load(input_ptr_count + (x0 + 144 * r1), r_mask & x_mask, other=0.0)
    running_mean = tl.load(input_ptr_running_mean + (x0), x_mask, eviction_policy='evict_last')
    running_var = tl.load(input_ptr_running_var + (x0), x_mask, eviction_policy='evict_last')

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

    # Reshape for further calculations
    reshaped_mean = mean[:, None]
    reshaped_var = var[:, None]

    # Compute inverse standard deviation
    epsilon = 1e-05
    normalized_var = reshaped_var / 36000.0
    adjusted_var = normalized_var + epsilon
    inv_std = tl.extra.cuda.libdevice.rsqrt(adjusted_var)

    # Update running statistics
    momentum = 0.1
    updated_mean = reshaped_mean * momentum
    decay = 0.9
    updated_running_mean = running_mean * decay + updated_mean

    scale_factor = 1.000027778549404
    scaled_var = reshaped_var * scale_factor
    scaled_updated_mean = scaled_var * momentum
    updated_running_var = running_var * decay + scaled_updated_mean

    # Store results
    tl.store(output_ptr_inv_std + (x0), inv_std, x_mask)
    tl.store(output_ptr_running_mean + (x0), updated_running_mean, x_mask)
    tl.store(output_ptr_running_var + (x0), updated_running_var, x_mask)
    tl.store(output_ptr_mean + (x0), reshaped_mean, x_mask)