# From: 52_Conv2d_Activation_BatchNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_mul_softplus_tanh_2(
    input_mean_ptr, input_var_ptr, input_gamma_ptr, input_beta_ptr, input_x_ptr,
    output_normalized_ptr, output_mean_ptr, output_var_ptr, output_beta_ptr, output_x_ptr,
    kernel_size_0, kernel_size_1, input_num_elements, running_num_elements, XBLOCK: tl.constexpr
):
    input_num_elements = 16
    running_num_elements = 15
    RBLOCK: tl.constexpr = 16
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < input_num_elements
    r_indices = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_indices < running_num_elements
    r1 = r_indices
    x0 = x_indices

    # Load input data with masks
    input_mean = tl.load(input_mean_ptr + (x0 + 16 * r1), r_mask & x_mask, other=0.0)
    input_var = tl.load(input_var_ptr + (x0 + 16 * r1), r_mask & x_mask, other=0.0)
    input_gamma = tl.load(input_gamma_ptr + (x0 + 16 * r1), r_mask & x_mask, other=0.0)
    input_beta = tl.load(input_beta_ptr + (x0), x_mask, eviction_policy='evict_last')
    input_x = tl.load(input_x_ptr + (x0), x_mask, eviction_policy='evict_last')

    # Broadcast loaded data
    broadcast_mean = tl.broadcast_to(input_mean, [XBLOCK, RBLOCK])
    broadcast_var = tl.broadcast_to(input_var, [XBLOCK, RBLOCK])
    broadcast_gamma = tl.broadcast_to(input_gamma, [XBLOCK, RBLOCK])

    # Apply masks
    masked_mean = tl.where(r_mask & x_mask, broadcast_mean, 0)
    masked_var = tl.where(r_mask & x_mask, broadcast_var, 0)
    masked_gamma = tl.where(r_mask & x_mask, broadcast_gamma, 0)

    # Compute Welford's algorithm for mean and variance
    mean, var, _ = triton_helpers.welford(masked_mean, masked_var, masked_gamma, 1)

    # Reshape mean and variance
    reshaped_mean = mean[:, None]
    reshaped_var = var[:, None]

    # Compute normalization parameters
    normalization_factor = 4 * kernel_size_0 + kernel_size_0 * kernel_size_1 * kernel_size_1 + ((-4) * kernel_size_0 * kernel_size_1)
    normalization_factor_float = normalization_factor.to(tl.float32)
    variance_adjustment = reshaped_var / normalization_factor_float
    epsilon = 1e-05
    adjusted_variance = variance_adjustment + epsilon
    reciprocal_sqrt = tl.extra.cuda.libdevice.rsqrt(adjusted_variance)

    # Compute scale and shift
    scale_factor = (((64 * kernel_size_0 + ((-64) * kernel_size_0 * kernel_size_1) + 16 * kernel_size_0 * kernel_size_1 * kernel_size_1) / 16) / 
                    ((tl.full([], -1.00000000000000, tl.float64)) + 
                     ((64 * kernel_size_0 + ((-64) * kernel_size_0 * kernel_size_1) + 16 * kernel_size_0 * kernel_size_1 * kernel_size_1) / 16)))
    scale_factor_float = scale_factor.to(tl.float32)
    scaled_variance = variance_adjustment * scale_factor_float
    momentum = 0.1
    scaled_momentum = scaled_variance * momentum
    momentum_beta = 0.9
    updated_beta = input_beta * momentum_beta
    new_beta = scaled_momentum + updated_beta
    momentum_gamma = 0.1
    scaled_gamma = reshaped_mean * momentum_gamma
    updated_gamma = input_gamma * momentum_beta
    new_gamma = scaled_gamma + updated_gamma

    # Store results
    tl.store(output_normalized_ptr + (x0), reciprocal_sqrt, x_mask)
    tl.store(output_beta_ptr + (x0), new_beta, x_mask)
    tl.store(output_var_ptr + (x0), new_gamma, x_mask)
    tl.store(output_mean_ptr + (x0), reshaped_mean, x_mask)
    tl.store(output_x_ptr + (x0), reshaped_var, x_mask)