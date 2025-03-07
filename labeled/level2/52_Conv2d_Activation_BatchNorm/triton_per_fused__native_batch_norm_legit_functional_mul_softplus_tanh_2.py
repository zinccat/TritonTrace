# From: 52_Conv2d_Activation_BatchNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_mul_softplus_tanh_2(
    input_ptr_mean, input_ptr_var, input_ptr_gamma, input_ptr_beta, input_ptr_x,
    output_ptr_normalized, output_ptr_mean, output_ptr_var, output_ptr_beta,
    output_ptr_x, xnumel, rnumel, XBLOCK: tl.constexpr
):
    xnumel = 16
    rnumel = 15
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex

    # Load inputs with masking
    input_mean = tl.load(input_ptr_mean + (x0 + (16 * r1)), rmask & xmask, other=0.0)
    input_var = tl.load(input_ptr_var + (x0 + (16 * r1)), rmask & xmask, other=0.0)
    input_gamma = tl.load(input_ptr_gamma + (x0 + (16 * r1)), rmask & xmask, other=0.0)
    input_beta = tl.load(input_ptr_beta + (x0), xmask, eviction_policy='evict_last')
    input_x = tl.load(input_ptr_x + (x0), xmask, eviction_policy='evict_last')

    # Broadcast inputs
    broadcast_mean = tl.broadcast_to(input_mean, [XBLOCK, RBLOCK])
    broadcast_var = tl.broadcast_to(input_var, [XBLOCK, RBLOCK])
    broadcast_gamma = tl.broadcast_to(input_gamma, [XBLOCK, RBLOCK])

    # Apply masks
    masked_mean = tl.where(rmask & xmask, broadcast_mean, 0)
    masked_var = tl.where(rmask & xmask, broadcast_var, 0)
    masked_gamma = tl.where(rmask & xmask, broadcast_gamma, 0)

    # Compute Welford's algorithm for mean and variance
    mean, var, _ = triton_helpers.welford(masked_mean, masked_var, masked_gamma, 1)

    # Normalize variance
    normalized_var = mean[:, None]
    variance_ratio = var[:, None] / 115200.0
    epsilon = 1e-05
    variance_ratio_eps = variance_ratio + epsilon
    inv_sqrt_var = tl.extra.cuda.libdevice.rsqrt(variance_ratio_eps)

    # Compute scale and shift
    scale_factor = 1.0000086806309083
    gamma_scale = variance_ratio * scale_factor
    gamma_scaled = gamma_scale * 0.1
    beta_scale = input_beta * 0.9
    scaled_beta = gamma_scaled + beta_scale

    x_scaled = mean * 0.1
    x_shifted = input_x * 0.9
    shifted_x = x_scaled + x_shifted

    # Store outputs
    tl.store(output_ptr_normalized + (x0), inv_sqrt_var, xmask)
    tl.store(output_ptr_beta + (x0), scaled_beta, xmask)
    tl.store(output_ptr_x + (x0), shifted_x, xmask)
    tl.store(output_ptr_mean + (x0), mean, xmask)
    tl.store(output_ptr_var + (x0), var, xmask)