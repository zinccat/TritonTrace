# From: 77_ConvTranspose3d_Scale_BatchNorm_GlobalAvgPool

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_mul_2(
    input_mean_ptr, input_var_ptr, input_x_ptr, input_beta_ptr, input_gamma_ptr,
    output_mean_ptr, output_var_ptr, output_x_ptr, output_beta_ptr, output_gamma_ptr,
    kernel_size_0, kernel_size_1, num_elements_x, num_elements_r, XBLOCK: tl.constexpr
):
    num_elements_x = 32
    num_elements_r = 11
    RBLOCK: tl.constexpr = 16
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < num_elements_x
    r_index = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_index < num_elements_r
    r1 = r_index
    x0 = x_index

    # Load input tensors
    input_mean = tl.load(input_mean_ptr + (x0 + 32 * r1), r_mask & x_mask, other=0.0)
    input_var = tl.load(input_var_ptr + (x0 + 32 * r1), r_mask & x_mask, other=0.0)
    input_x = tl.load(input_x_ptr + (x0 + 32 * r1), r_mask & x_mask, other=0.0)
    input_beta = tl.load(input_beta_ptr + (x0), x_mask, eviction_policy='evict_last')
    input_gamma = tl.load(input_gamma_ptr + (x0), x_mask, eviction_policy='evict_last')

    # Broadcast loaded values
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
    normalization_factor = 4 * kernel_size_0 * kernel_size_0 + 8 * kernel_size_0 + \
                           kernel_size_0 * kernel_size_0 * kernel_size_1 * kernel_size_1 + \
                           2 * kernel_size_0 * kernel_size_1 * kernel_size_1 + \
                           4 * kernel_size_1 * kernel_size_0 * kernel_size_0 + 8 * kernel_size_0 * kernel_size_1
    normalization_factor_float = normalization_factor.to(tl.float32)
    variance_adjustment = reshaped_var / normalization_factor_float
    epsilon = 1e-05
    adjusted_variance = variance_adjustment + epsilon
    inv_stddev = tl.extra.cuda.libdevice.rsqrt(adjusted_variance)

    # Compute scale factor
    scale_factor = (((128 * kernel_size_0 * kernel_size_0 + 256 * kernel_size_0 + \
                      32 * kernel_size_0 * kernel_size_0 * kernel_size_1 * kernel_size_1 + \
                      64 * kernel_size_0 * kernel_size_1 * kernel_size_1 + \
                      128 * kernel_size_1 * kernel_size_0 * kernel_size_0 + 256 * kernel_size_0 * kernel_size_1) / 32) / \
                     ((tl.full([], -1.00000000000000, tl.float64)) + \
                      ((128 * kernel_size_0 * kernel_size_0 + 256 * kernel_size_0 + \
                        32 * kernel_size_0 * kernel_size_0 * kernel_size_1 * kernel_size_1 + \
                        64 * kernel_size_0 * kernel_size_1 * kernel_size_1 + \
                        128 * kernel_size_1 * kernel_size_0 * kernel_size_0 + 256 * kernel_size_0 * kernel_size_1) / 32)))
    scale_factor_float = scale_factor.to(tl.float32)
    scaled_variance = variance_adjustment * scale_factor_float

    # Compute running mean and variance
    momentum = 0.1
    running_mean = scaled_variance * momentum
    running_mean_factor = reshaped_mean * momentum
    running_var_factor = input_beta * 0.9
    updated_running_mean = running_mean + running_mean_factor
    updated_running_var = running_var_factor + running_var_factor

    # Store results
    tl.store(output_x_ptr + (x0), inv_stddev, x_mask)
    tl.store(output_beta_ptr + (x0), updated_running_mean, x_mask)
    tl.store(output_gamma_ptr + (x0), updated_running_var, x_mask)
    tl.store(output_mean_ptr + (x0), reshaped_mean, x_mask)
    tl.store(output_var_ptr + (x0), reshaped_var, x_mask)