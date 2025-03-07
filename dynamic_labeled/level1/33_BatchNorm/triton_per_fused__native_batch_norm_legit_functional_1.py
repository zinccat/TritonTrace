# From: 33_BatchNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_1(
    input_mean_ptr, input_var_ptr, input_gamma_ptr, input_beta_ptr, input_x_ptr,
    output_mean_ptr, output_var_ptr, output_x_ptr, output_beta_ptr, output_gamma_ptr,
    kernel_size_0, kernel_size_1, num_elements_x, num_elements_r, XBLOCK: tl.constexpr
):
    num_elements_x = 64
    num_elements_r = 6
    RBLOCK: tl.constexpr = 8
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements_x
    r_indices = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_indices < num_elements_r
    r1 = r_indices
    x0 = x_indices

    mean_values = tl.load(input_mean_ptr + (x0 + 64 * r1), r_mask & x_mask, other=0.0)
    var_values = tl.load(input_var_ptr + (x0 + 64 * r1), r_mask & x_mask, other=0.0)
    gamma_values = tl.load(input_gamma_ptr + (x0 + 64 * r1), r_mask & x_mask, other=0.0)
    beta_values = tl.load(input_beta_ptr + (x0), x_mask, eviction_policy='evict_last')
    x_values = tl.load(input_x_ptr + (x0), x_mask, eviction_policy='evict_last')

    mean_broadcast = tl.broadcast_to(mean_values, [XBLOCK, RBLOCK])
    var_broadcast = tl.broadcast_to(var_values, [XBLOCK, RBLOCK])
    gamma_broadcast = tl.broadcast_to(gamma_values, [XBLOCK, RBLOCK])

    mean_selected = tl.where(r_mask & x_mask, mean_broadcast, 0)
    var_selected = tl.where(r_mask & x_mask, var_broadcast, 0)
    gamma_selected = tl.where(r_mask & x_mask, gamma_broadcast, 0)

    mean_accum, var_accum, count = triton_helpers.welford(mean_selected, var_selected, gamma_selected, 1)
    mean_accum_expanded = mean_accum[:, None]
    var_accum_expanded = var_accum[:, None]

    num_elements = kernel_size_0 * kernel_size_1 * kernel_size_1
    num_elements_float = num_elements.to(tl.float32)
    mean_normalized = var_accum_expanded / num_elements_float

    epsilon = 1e-05
    mean_normalized_eps = mean_normalized + epsilon
    inv_std = tl.extra.cuda.libdevice.rsqrt(mean_normalized_eps)

    factor = ((64 * kernel_size_0 * kernel_size_1 * kernel_size_1) / 64) / (
        (tl.full([], -1.0, tl.float64)) + ((64 * kernel_size_0 * kernel_size_1 * kernel_size_1) / 64)
    )
    factor_float = factor.to(tl.float32)
    var_scaled = mean_normalized * factor_float

    momentum = 0.1
    var_momentum = var_scaled * momentum
    beta_momentum = beta_values * 0.9
    updated_beta = var_momentum + beta_momentum

    gamma_momentum = mean_accum_expanded * momentum
    updated_gamma = gamma_momentum + (gamma_values * 0.9)

    tl.store(output_x_ptr + (x0), inv_std, x_mask)
    tl.store(output_beta_ptr + (x0), updated_beta, x_mask)
    tl.store(output_gamma_ptr + (x0), updated_gamma, x_mask)
    tl.store(output_mean_ptr + (x0), mean_accum_expanded, x_mask)
    tl.store(output_var_ptr + (x0), var_accum_expanded, x_mask)