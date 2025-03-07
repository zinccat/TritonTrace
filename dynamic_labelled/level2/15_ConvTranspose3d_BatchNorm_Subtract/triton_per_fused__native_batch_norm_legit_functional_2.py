# From: 15_ConvTranspose3d_BatchNorm_Subtract

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_2(
    input_mean_ptr, input_var_ptr, input_x_ptr, running_mean_ptr, running_var_ptr,
    output_mean_ptr, output_var_ptr, output_x_ptr, output_running_mean_ptr, output_running_var_ptr,
    kernel_size, num_elements, reduced_num_elements, XBLOCK: tl.constexpr
):
    num_elements = 32
    reduced_num_elements = 11
    RBLOCK: tl.constexpr = 16
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements
    r_indices = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_indices < reduced_num_elements
    r1 = r_indices
    x0 = x_indices

    # Load input data with masks
    input_mean = tl.load(input_mean_ptr + (x0 + 32 * r1), r_mask & x_mask, other=0.0)
    input_var = tl.load(input_var_ptr + (x0 + 32 * r1), r_mask & x_mask, other=0.0)
    input_x = tl.load(input_x_ptr + (x0 + 32 * r1), r_mask & x_mask, other=0.0)
    running_mean = tl.load(running_mean_ptr + (x0), x_mask, eviction_policy='evict_last')
    running_var = tl.load(running_var_ptr + (x0), x_mask, eviction_policy='evict_last')

    # Broadcast loaded data
    broadcast_mean = tl.broadcast_to(input_mean, [XBLOCK, RBLOCK])
    broadcast_var = tl.broadcast_to(input_var, [XBLOCK, RBLOCK])
    broadcast_x = tl.broadcast_to(input_x, [XBLOCK, RBLOCK])

    # Apply masks
    masked_mean = tl.where(r_mask & x_mask, broadcast_mean, 0)
    masked_var = tl.where(r_mask & x_mask, broadcast_var, 0)
    masked_x = tl.where(r_mask & x_mask, broadcast_x, 0)

    # Compute Welford's algorithm for mean and variance
    mean, var, _ = triton_helpers.welford(masked_mean, masked_var, masked_x, 1)

    # Reshape mean and variance
    reshaped_mean = mean[:, None]
    reshaped_var = var[:, None]

    # Compute normalization parameters
    factor = 496 + ((-1984) * kernel_size) + 1984 * kernel_size * kernel_size
    factor_float = factor.to(tl.float32)
    normalized_var = reshaped_var / factor_float
    epsilon = 1e-05
    adjusted_var = normalized_var + epsilon
    inv_std = tl.extra.cuda.libdevice.rsqrt(adjusted_var)

    # Compute running variance adjustment
    adjustment_factor = (((15872 + ((-63488) * kernel_size) + 63488 * kernel_size * kernel_size) / 32) /
                         ((tl.full([], -1.00000000000000, tl.float64)) + 
                         ((15872 + ((-63488) * kernel_size) + 63488 * kernel_size * kernel_size) / 32)))
    adjustment_factor_float = adjustment_factor.to(tl.float32)
    var_adjustment = normalized_var * adjustment_factor_float
    momentum = 0.1
    var_momentum = var_adjustment * momentum
    momentum_factor = 0.9
    updated_running_mean = running_mean * momentum_factor + var_momentum
    updated_running_var = reshaped_mean * momentum + running_var * momentum_factor

    # Store results
    tl.store(output_mean_ptr + (x0), inv_std, x_mask)
    tl.store(output_var_ptr + (x0), updated_running_mean, x_mask)
    tl.store(output_x_ptr + (x0), updated_running_var, x_mask)
    tl.store(output_running_mean_ptr + (x0), reshaped_mean, x_mask)
    tl.store(output_running_var_ptr + (x0), reshaped_var, x_mask)