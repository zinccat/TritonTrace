# From: 52_Conv2d_Activation_BatchNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_mul_softplus_tanh_2(
    input_mean_ptr, input_var_ptr, input_gamma_ptr, input_beta_ptr, input_x_ptr,
    output_mean_ptr, output_var_ptr, output_normalized_ptr, output_beta_ptr, output_gamma_ptr,
    kernel_size_0, kernel_size_1, input_num_elements, running_num_elements,
    XBLOCK: tl.constexpr
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

    mean = tl.load(input_mean_ptr + (x0 + 16 * r1), r_mask & x_mask, other=0.0)
    var = tl.load(input_var_ptr + (x0 + 16 * r1), r_mask & x_mask, other=0.0)
    gamma = tl.load(input_gamma_ptr + (x0 + 16 * r1), r_mask & x_mask, other=0.0)
    beta_running_mean = tl.load(input_beta_ptr + (x0), x_mask, eviction_policy='evict_last')
    beta_running_var = tl.load(input_x_ptr + (x0), x_mask, eviction_policy='evict_last')

    mean_broadcast = tl.broadcast_to(mean, [XBLOCK, RBLOCK])
    var_broadcast = tl.broadcast_to(var, [XBLOCK, RBLOCK])
    gamma_broadcast = tl.broadcast_to(gamma, [XBLOCK, RBLOCK])

    mean_selected = tl.where(r_mask & x_mask, mean_broadcast, 0)
    var_selected = tl.where(r_mask & x_mask, var_broadcast, 0)
    gamma_selected = tl.where(r_mask & x_mask, gamma_broadcast, 0)

    mean_accum, var_accum, count = triton_helpers.welford(mean_selected, var_selected, gamma_selected, 1)
    mean_accum_expanded = mean_accum[:, None]
    var_accum_expanded = var_accum[:, None]

    normalization_factor = 4 * kernel_size_0 + kernel_size_0 * kernel_size_1 * kernel_size_1 + ((-4) * kernel_size_0 * kernel_size_1)
    normalization_factor_float = normalization_factor.to(tl.float32)

    var_normalized = var_accum_expanded / normalization_factor_float
    epsilon = 1e-05
    var_normalized_eps = var_normalized + epsilon
    inv_sqrt_var = tl.extra.cuda.libdevice.rsqrt(var_normalized_eps)

    factor = (((64 * kernel_size_0 + ((-64) * kernel_size_0 * kernel_size_1) + 16 * kernel_size_0 * kernel_size_1 * kernel_size_1) / 16) /
              ((tl.full([], -1.00000000000000, tl.float64)) + ((64 * kernel_size_0 + ((-64) * kernel_size_0 * kernel_size_1) + 16 * kernel_size_0 * kernel_size_1 * kernel_size_1) / 16)))
    factor_float = factor.to(tl.float32)
    var_scaled = var_normalized * factor_float

    momentum = 0.1
    var_momentum = var_scaled * momentum
    momentum_beta = 0.9
    beta_running_mean_momentum = beta_running_mean * momentum_beta
    beta_updated = var_momentum + beta_running_mean_momentum

    gamma_momentum = mean_accum_expanded * momentum
    gamma_running_var_momentum = beta_running_var * momentum_beta
    gamma_updated = gamma_momentum + gamma_running_var_momentum

    tl.store(output_var_ptr + (x0), inv_sqrt_var, x_mask)
    tl.store(output_beta_ptr + (x0), beta_updated, x_mask)
    tl.store(output_gamma_ptr + (x0), gamma_updated, x_mask)
    tl.store(output_mean_ptr + (x0), mean_accum_expanded, x_mask)
    tl.store(output_var_ptr + (x0), var_accum_expanded, x_mask)