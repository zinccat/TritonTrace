# From: 23_EfficientNetB1

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_6(
    input_ptr_mean, input_ptr_var, input_ptr_count, input_ptr_running_mean, input_ptr_running_var,
    output_ptr_inv_std, output_ptr_running_mean, output_ptr_running_var, output_ptr_mean,
    xnumel, rnumel, XBLOCK: tl.constexpr
):
    xnumel = 32
    rnumel = 9
    RBLOCK: tl.constexpr = 16
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_index = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_index < rnumel
    r1 = r_index
    x0 = x_index

    # Load input data
    input_mean = tl.load(input_ptr_mean + (x0 + 32 * r1), r_mask & x_mask, other=0.0)
    input_var = tl.load(input_ptr_var + (x0 + 32 * r1), r_mask & x_mask, other=0.0)
    input_count = tl.load(input_ptr_count + (x0 + 32 * r1), r_mask & x_mask, other=0.0)
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
    adjusted_var = reshaped_var / 144000.0 + epsilon
    inv_std = tl.extra.cuda.libdevice.rsqrt(adjusted_var)

    # Update running mean and variance
    momentum = 0.1
    updated_running_mean = reshaped_mean * momentum + running_mean * 0.9
    updated_running_var = reshaped_var * 1.00000694449267 * momentum + running_var * 0.9

    # Store results
    tl.store(output_ptr_inv_std + (x0), inv_std, x_mask)
    tl.store(output_ptr_running_mean + (x0), updated_running_mean, x_mask)
    tl.store(output_ptr_running_var + (x0), updated_running_var, x_mask)
    tl.store(output_ptr_mean + (x0), reshaped_mean, x_mask)