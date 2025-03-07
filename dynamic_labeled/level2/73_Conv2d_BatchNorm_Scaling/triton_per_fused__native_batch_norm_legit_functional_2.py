# From: 73_Conv2d_BatchNorm_Scaling

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_2(
    input_ptr_mean, input_ptr_var, input_ptr_count, input_ptr_gamma, input_ptr_beta,
    output_ptr_normalized, output_ptr_mean, output_ptr_var, output_ptr_gamma, output_ptr_beta,
    kernel_size_0, kernel_size_1, num_elements_x, num_elements_r, XBLOCK: tl.constexpr
):
    num_elements_x = 16
    num_elements_r = 15
    RBLOCK: tl.constexpr = 16
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < num_elements_x
    r_index = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_index < num_elements_r
    r1 = r_index
    x0 = x_index

    mean_accumulator = tl.load(input_ptr_mean + (x0 + 16 * r1), r_mask & x_mask, other=0.0)
    var_accumulator = tl.load(input_ptr_var + (x0 + 16 * r1), r_mask & x_mask, other=0.0)
    count_accumulator = tl.load(input_ptr_count + (x0 + 16 * r1), r_mask & x_mask, other=0.0)
    gamma = tl.load(input_ptr_gamma + (x0), x_mask, eviction_policy='evict_last')
    beta = tl.load(input_ptr_beta + (x0), x_mask, eviction_policy='evict_last')

    mean_broadcast = tl.broadcast_to(mean_accumulator, [XBLOCK, RBLOCK])
    var_broadcast = tl.broadcast_to(var_accumulator, [XBLOCK, RBLOCK])
    count_broadcast = tl.broadcast_to(count_accumulator, [XBLOCK, RBLOCK])

    mean_selected = tl.where(r_mask & x_mask, mean_broadcast, 0)
    var_selected = tl.where(r_mask & x_mask, var_broadcast, 0)
    count_selected = tl.where(r_mask & x_mask, count_broadcast, 0)

    mean_welford, var_welford, count_welford = triton_helpers.welford(
        mean_selected, var_selected, count_selected, 1
    )

    mean_welford_expanded = mean_welford[:, None]
    var_welford_expanded = var_welford[:, None]

    normalization_factor = 4 * kernel_size_0 + kernel_size_0 * kernel_size_1 * kernel_size_1 + ((-4) * kernel_size_0 * kernel_size_1)
    normalization_factor_float = normalization_factor.to(tl.float32)

    var_normalized = var_welford_expanded / normalization_factor_float
    epsilon = 1e-05
    var_normalized_eps = var_normalized + epsilon

    inv_sqrt_var = tl.extra.cuda.libdevice.rsqrt(var_normalized_eps)

    factor = (((64 * kernel_size_0 + ((-64) * kernel_size_0 * kernel_size_1) + 16 * kernel_size_0 * kernel_size_1 * kernel_size_1) / 16) /
              ((tl.full([], -1.00000000000000, tl.float64)) + ((64 * kernel_size_0 + ((-64) * kernel_size_0 * kernel_size_1) + 16 * kernel_size_0 * kernel_size_1 * kernel_size_1) / 16)))
    factor_float = factor.to(tl.float32)

    var_scaled = var_normalized * factor_float
    momentum = 0.1
    var_momentum = var_scaled * momentum

    gamma_momentum = 0.9
    gamma_updated = gamma * gamma_momentum
    gamma_new = var_momentum + gamma_updated

    beta_momentum = 0.9
    beta_updated = beta * beta_momentum
    beta_new = (mean_welford_expanded * momentum) + beta_updated

    tl.store(output_ptr_normalized + (x0), inv_sqrt_var, x_mask)
    tl.store(output_ptr_gamma + (x0), gamma_new, x_mask)
    tl.store(output_ptr_beta + (x0), beta_new, x_mask)
    tl.store(output_ptr_mean + (x0), mean_welford_expanded, x_mask)
    tl.store(output_ptr_var + (x0), var_welford_expanded, x_mask)