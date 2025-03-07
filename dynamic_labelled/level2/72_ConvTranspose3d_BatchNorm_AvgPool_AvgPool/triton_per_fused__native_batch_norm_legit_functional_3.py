# From: 72_ConvTranspose3d_BatchNorm_AvgPool_AvgPool

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_3(
    input_ptr_mean, input_ptr_var, input_ptr_count, input_ptr_beta, input_ptr_gamma,
    output_ptr_normalized, output_ptr_mean, output_ptr_var, output_ptr_beta, output_ptr_gamma,
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

    mean_accumulator = tl.load(input_ptr_mean + (r1 + 249 * x0), r_mask & x_mask, other=0.0)
    var_accumulator = tl.load(input_ptr_var + (r1 + 249 * x0), r_mask & x_mask, other=0.0)
    count_accumulator = tl.load(input_ptr_count + (r1 + 249 * x0), r_mask & x_mask, other=0.0)
    beta = tl.load(input_ptr_beta + (x0), x_mask, eviction_policy='evict_last')
    gamma = tl.load(input_ptr_gamma + (x0), x_mask, eviction_policy='evict_last')

    mean_broadcast = tl.broadcast_to(mean_accumulator, [XBLOCK, RBLOCK])
    var_broadcast = tl.broadcast_to(var_accumulator, [XBLOCK, RBLOCK])
    count_broadcast = tl.broadcast_to(count_accumulator, [XBLOCK, RBLOCK])

    mean_selected = tl.where(r_mask & x_mask, mean_broadcast, 0)
    var_selected = tl.where(r_mask & x_mask, var_broadcast, 0)
    count_selected = tl.where(r_mask & x_mask, count_broadcast, 0)

    mean_moments, var_moments, count_moments = triton_helpers.welford(
        mean_selected, var_selected, count_selected, 1
    )

    mean_moments_expanded = mean_moments[:, None]
    var_moments_expanded = var_moments[:, None]

    normalization_factor = ((-1) * kernel_size_0) + ((-12) * kernel_size_0 * kernel_size_1 * kernel_size_1) + 6 * kernel_size_0 * kernel_size_1 + 8 * kernel_size_0 * kernel_size_1 * kernel_size_1 * kernel_size_1
    normalization_factor_float = normalization_factor.to(tl.float32)

    variance_normalized = var_moments_expanded / normalization_factor_float
    epsilon = 1e-05
    variance_normalized_eps = variance_normalized + epsilon
    rsqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_normalized_eps)

    scale_factor = (((((-16) * kernel_size_0) + ((-192) * kernel_size_0 * kernel_size_1 * kernel_size_1) + 96 * kernel_size_0 * kernel_size_1 + 128 * kernel_size_0 * kernel_size_1 * kernel_size_1 * kernel_size_1) / 16) / ((tl.full([], -1.00000000000000, tl.float64)) + ((((-16) * kernel_size_0) + ((-192) * kernel_size_0 * kernel_size_1 * kernel_size_1) + 96 * kernel_size_0 * kernel_size_1 + 128 * kernel_size_0 * kernel_size_1 * kernel_size_1 * kernel_size_1) / 16)))
    scale_factor_float = scale_factor.to(tl.float32)

    variance_scaled = variance_normalized * scale_factor_float
    momentum = 0.1
    variance_momentum = variance_scaled * momentum
    momentum_beta = 0.9
    beta_momentum = beta * momentum_beta
    updated_beta = variance_momentum + beta_momentum

    gamma_momentum = gamma * momentum_beta
    updated_gamma = mean_moments_expanded * momentum + gamma_momentum

    tl.store(output_ptr_normalized + (x0), rsqrt_variance, x_mask)
    tl.store(output_ptr_mean + (x0), updated_beta, x_mask)
    tl.store(output_ptr_var + (x0), updated_gamma, x_mask)
    tl.store(output_ptr_beta + (x0), mean_moments_expanded, x_mask)
    tl.store(output_ptr_gamma + (x0), var_moments_expanded, x_mask)