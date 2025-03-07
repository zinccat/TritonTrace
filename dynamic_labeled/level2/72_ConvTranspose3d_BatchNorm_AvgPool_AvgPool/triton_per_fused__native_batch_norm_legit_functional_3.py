# From: 72_ConvTranspose3d_BatchNorm_AvgPool_AvgPool

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_3(
    input_ptr_mean, input_ptr_var, input_ptr_count, input_ptr_beta, input_ptr_gamma,
    output_ptr_mean, output_ptr_var, output_ptr_beta, output_ptr_gamma, output_ptr_output,
    kernel_size_0, kernel_size_1, num_elements_x, num_elements_r, XBLOCK: tl.constexpr
):
    num_elements_x = 16
    num_elements_r = 249
    RBLOCK: tl.constexpr = 256
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < num_elements_x
    r_index = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_index < num_elements_r
    r1 = r_index
    x0 = x_index

    mean_input = tl.load(input_ptr_mean + (r1 + 249 * x0), r_mask & x_mask, other=0.0)
    var_input = tl.load(input_ptr_var + (r1 + 249 * x0), r_mask & x_mask, other=0.0)
    count_input = tl.load(input_ptr_count + (r1 + 249 * x0), r_mask & x_mask, other=0.0)

    beta_input = tl.load(input_ptr_beta + (x0), x_mask, eviction_policy='evict_last')
    gamma_input = tl.load(input_ptr_gamma + (x0), x_mask, eviction_policy='evict_last')

    mean_broadcast = tl.broadcast_to(mean_input, [XBLOCK, RBLOCK])
    var_broadcast = tl.broadcast_to(var_input, [XBLOCK, RBLOCK])
    count_broadcast = tl.broadcast_to(count_input, [XBLOCK, RBLOCK])

    mean_masked = tl.where(r_mask & x_mask, mean_broadcast, 0)
    var_masked = tl.where(r_mask & x_mask, var_broadcast, 0)
    count_masked = tl.where(r_mask & x_mask, count_broadcast, 0)

    mean_accum, var_accum, count_accum = triton_helpers.welford(mean_masked, var_masked, count_masked, 1)

    mean_accum_expanded = mean_accum[:, None]
    var_accum_expanded = var_accum[:, None]

    normalization_factor = ((-1) * kernel_size_0) + ((-12) * kernel_size_0 * kernel_size_1 * kernel_size_1) + 6 * kernel_size_0 * kernel_size_1 + 8 * kernel_size_0 * kernel_size_1 * kernel_size_1 * kernel_size_1
    normalization_factor_float = normalization_factor.to(tl.float32)

    var_normalized = var_accum_expanded / normalization_factor_float
    epsilon = 1e-05
    var_normalized_eps = var_normalized + epsilon

    inv_sqrt_var = tl.extra.cuda.libdevice.rsqrt(var_normalized_eps)

    correction_factor = (((((-16) * kernel_size_0) + ((-192) * kernel_size_0 * kernel_size_1 * kernel_size_1) + 96 * kernel_size_0 * kernel_size_1 + 128 * kernel_size_0 * kernel_size_1 * kernel_size_1 * kernel_size_1) / 16) / ((tl.full([], -1.00000000000000, tl.float64)) + ((((-16) * kernel_size_0) + ((-192) * kernel_size_0 * kernel_size_1 * kernel_size_1) + 96 * kernel_size_0 * kernel_size_1 + 128 * kernel_size_0 * kernel_size_1 * kernel_size_1 * kernel_size_1) / 16)))
    correction_factor_float = correction_factor.to(tl.float32)

    var_corrected = var_normalized * correction_factor_float
    momentum = 0.1
    var_momentum = var_corrected * momentum
    momentum_beta = 0.9
    beta_momentum = beta_input * momentum_beta
    beta_updated = var_momentum + beta_momentum

    gamma_momentum = mean_accum_expanded * momentum
    gamma_momentum_beta = gamma_input * momentum_beta
    gamma_updated = gamma_momentum + gamma_momentum_beta

    tl.store(output_ptr_var + (x0), inv_sqrt_var, x_mask)
    tl.store(output_ptr_beta + (x0), beta_updated, x_mask)
    tl.store(output_ptr_gamma + (x0), gamma_updated, x_mask)
    tl.store(output_ptr_mean + (x0), mean_accum_expanded, x_mask)
    tl.store(output_ptr_output + (x0), var_accum_expanded, x_mask)