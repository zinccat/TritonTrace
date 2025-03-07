# From: 73_Conv2d_BatchNorm_Scaling

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_2(
    input_ptr_mean, input_ptr_var, input_ptr_gamma, input_ptr_beta, input_ptr_input,
    output_ptr_normalized, output_ptr_mean, output_ptr_var, output_ptr_gamma, output_ptr_beta,
    num_elements, num_features, XBLOCK: tl.constexpr
):
    num_elements = 16
    num_features = 15
    RBLOCK: tl.constexpr = 16
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements
    r_indices = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_indices < num_features
    r_feature_indices = r_indices
    x_input_indices = x_indices

    mean_values = tl.load(input_ptr_mean + (x_input_indices + (16 * r_feature_indices)), r_mask & x_mask, other=0.0)
    var_values = tl.load(input_ptr_var + (x_input_indices + (16 * r_feature_indices)), r_mask & x_mask, other=0.0)
    gamma_values = tl.load(input_ptr_gamma + (x_input_indices + (16 * r_feature_indices)), r_mask & x_mask, other=0.0)
    input_values = tl.load(input_ptr_input + (x_input_indices), x_mask, eviction_policy='evict_last')
    beta_values = tl.load(input_ptr_beta + (x_input_indices), x_mask, eviction_policy='evict_last')

    mean_broadcast = tl.broadcast_to(mean_values, [XBLOCK, RBLOCK])
    var_broadcast = tl.broadcast_to(var_values, [XBLOCK, RBLOCK])
    gamma_broadcast = tl.broadcast_to(gamma_values, [XBLOCK, RBLOCK])

    mean_selected = tl.where(r_mask & x_mask, mean_broadcast, 0)
    var_selected = tl.where(r_mask & x_mask, var_broadcast, 0)
    gamma_selected = tl.where(r_mask & x_mask, gamma_broadcast, 0)

    mean_accum, var_accum, count = triton_helpers.welford(mean_selected, var_selected, gamma_selected, 1)
    mean_accum_broadcast = mean_accum[:, None]
    var_accum_broadcast = var_accum[:, None]

    epsilon = 1e-05
    rsqrt_var = tl.extra.cuda.libdevice.rsqrt(var_accum_broadcast / 115200.0 + epsilon)
    gamma_scaled = var_accum_broadcast / 115200.0 * 1.0000086806309083 * 0.1
    beta_scaled = input_values * 0.9 + gamma_scaled * 0.1
    mean_scaled = mean_accum_broadcast * 0.1

    tl.store(output_ptr_normalized + (x_input_indices), rsqrt_var, x_mask)
    tl.store(output_ptr_gamma + (x_input_indices), gamma_scaled + input_values * 0.9, x_mask)
    tl.store(output_ptr_beta + (x_input_indices), beta_scaled, x_mask)
    tl.store(output_ptr_mean + (x_input_indices), mean_scaled, x_mask)
    tl.store(output_ptr_var + (x_input_indices), var_accum_broadcast, x_mask)