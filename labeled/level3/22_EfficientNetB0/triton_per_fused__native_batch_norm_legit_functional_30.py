# From: 22_EfficientNetB0

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_30(
    input_ptr_mean, input_ptr_var, input_ptr_count, input_ptr_running_mean, input_ptr_running_var,
    output_ptr_inv_std, output_ptr_running_mean, output_ptr_running_var, output_ptr_mean, 
    num_elements, num_running_elements, XBLOCK: tl.constexpr
):
    num_elements = 144
    num_running_elements = 62
    RBLOCK: tl.constexpr = 64
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements
    r_indices = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_indices < num_running_elements
    running_indices = r_indices
    input_indices = x_indices

    mean = tl.load(input_ptr_mean + (input_indices + 144 * running_indices), r_mask & x_mask, other=0.0)
    variance = tl.load(input_ptr_var + (input_indices + 144 * running_indices), r_mask & x_mask, other=0.0)
    count = tl.load(input_ptr_count + (input_indices + 144 * running_indices), r_mask & x_mask, other=0.0)
    running_mean = tl.load(input_ptr_running_mean + (input_indices), x_mask, eviction_policy='evict_last')
    running_var = tl.load(input_ptr_running_var + (input_indices), x_mask, eviction_policy='evict_last')

    mean_broadcast = tl.broadcast_to(mean, [XBLOCK, RBLOCK])
    variance_broadcast = tl.broadcast_to(variance, [XBLOCK, RBLOCK])
    count_broadcast = tl.broadcast_to(count, [XBLOCK, RBLOCK])

    mean_masked = tl.where(r_mask & x_mask, mean_broadcast, 0)
    variance_masked = tl.where(r_mask & x_mask, variance_broadcast, 0)
    count_masked = tl.where(r_mask & x_mask, count_broadcast, 0)

    mean_accum, variance_accum, count_accum = triton_helpers.welford(mean_masked, variance_masked, count_masked, 1)

    mean_accum_expanded = mean_accum[:, None]
    variance_accum_expanded = variance_accum[:, None]

    normalization_factor = 7840.0
    epsilon = 1e-05
    adjusted_variance = variance_accum_expanded / normalization_factor
    adjusted_variance_with_epsilon = adjusted_variance + epsilon

    inv_std = tl.extra.cuda.libdevice.rsqrt(adjusted_variance_with_epsilon)

    momentum = 0.1
    mean_scaled = mean_accum_expanded * momentum

    running_mean_momentum = 0.9
    updated_running_mean = running_mean * running_mean_momentum
    new_running_mean = mean_scaled + updated_running_mean

    bias_correction = 1.0001275672917465
    variance_scaled = adjusted_variance * bias_correction
    variance_scaled_with_momentum = variance_scaled * momentum

    updated_running_var = running_var * running_mean_momentum
    new_running_var = variance_scaled_with_momentum + updated_running_var

    tl.store(output_ptr_inv_std + (input_indices), inv_std, x_mask)
    tl.store(output_ptr_running_mean + (input_indices), new_running_mean, x_mask)
    tl.store(output_ptr_running_var + (input_indices), new_running_var, x_mask)
    tl.store(output_ptr_mean + (input_indices), mean_accum_expanded, x_mask)