# From: 77_ConvTranspose3d_Scale_BatchNorm_GlobalAvgPool

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_mul_2(
    input_ptr_mean, input_ptr_var, input_ptr_gamma, input_ptr_beta, input_ptr_input,
    output_ptr_normalized, output_ptr_mean, output_ptr_var, output_ptr_beta, output_ptr_input,
    num_elements, num_reduction_elements, XBLOCK: tl.constexpr
):
    num_elements = 32
    num_reduction_elements = 11
    RBLOCK: tl.constexpr = 16
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements
    r_indices = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_indices < num_reduction_elements
    r1 = r_indices
    x0 = x_indices

    mean = tl.load(input_ptr_mean + (x0 + (32 * r1)), r_mask & x_mask, other=0.0)
    var = tl.load(input_ptr_var + (x0 + (32 * r1)), r_mask & x_mask, other=0.0)
    gamma = tl.load(input_ptr_gamma + (x0 + (32 * r1)), r_mask & x_mask, other=0.0)
    input_data = tl.load(input_ptr_input + (x0), x_mask, eviction_policy='evict_last')
    beta = tl.load(input_ptr_beta + (x0), x_mask, eviction_policy='evict_last')

    mean_broadcast = tl.broadcast_to(mean, [XBLOCK, RBLOCK])
    var_broadcast = tl.broadcast_to(var, [XBLOCK, RBLOCK])
    gamma_broadcast = tl.broadcast_to(gamma, [XBLOCK, RBLOCK])

    mean_selected = tl.where(r_mask & x_mask, mean_broadcast, 0)
    var_selected = tl.where(r_mask & x_mask, var_broadcast, 0)
    gamma_selected = tl.where(r_mask & x_mask, gamma_broadcast, 0)

    mean_accum, var_accum, count = triton_helpers.welford(mean_selected, var_selected, gamma_selected, 1)
    mean_accum_expanded = mean_accum[:, None]
    var_accum_expanded = var_accum[:, None]

    epsilon = 1e-05
    variance_adjusted = var_accum_expanded / 332928.0
    variance_adjusted += epsilon
    inv_stddev = tl.extra.cuda.libdevice.rsqrt(variance_adjusted)

    gamma_factor = 1.0000030036614633
    gamma_scaled = variance_adjusted * gamma_factor
    momentum = 0.1
    gamma_momentum = gamma_scaled * momentum

    beta_momentum = 0.9
    beta_accumulated = input_data * beta_momentum
    beta_updated = gamma_momentum + beta_accumulated

    beta_scaled = mean_accum_expanded * momentum
    beta_final = beta_scaled + beta_accumulated * beta_momentum

    tl.store(output_ptr_normalized + (x0), inv_stddev, x_mask)
    tl.store(output_ptr_beta + (x0), beta_updated, x_mask)
    tl.store(output_ptr_input + (x0), beta_final, x_mask)
    tl.store(output_ptr_mean + (x0), mean_accum_expanded, x_mask)
    tl.store(output_ptr_var + (x0), var_accum_expanded, x_mask)