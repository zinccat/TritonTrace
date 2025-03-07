# From: 23_EfficientNetB1

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_53(
    input_ptr_mean, input_ptr_var, input_ptr_count, input_ptr_running_mean, input_ptr_running_var,
    output_ptr_normalized, output_ptr_running_mean, output_ptr_running_var, output_ptr_mean,
    num_elements, num_running_stats, XBLOCK: tl.constexpr
):
    num_elements = 672
    num_running_stats = 5
    RBLOCK: tl.constexpr = 8
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements
    r_indices = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_indices < num_running_stats
    running_indices = r_indices
    input_indices = x_indices

    mean_values = tl.load(input_ptr_mean + (input_indices + 672 * running_indices), r_mask & x_mask, other=0.0)
    var_values = tl.load(input_ptr_var + (input_indices + 672 * running_indices), r_mask & x_mask, other=0.0)
    count_values = tl.load(input_ptr_count + (input_indices + 672 * running_indices), r_mask & x_mask, other=0.0)
    running_mean_values = tl.load(input_ptr_running_mean + (input_indices), x_mask, eviction_policy='evict_last')
    running_var_values = tl.load(input_ptr_running_var + (input_indices), x_mask, eviction_policy='evict_last')

    broadcast_mean = tl.broadcast_to(mean_values, [XBLOCK, RBLOCK])
    broadcast_var = tl.broadcast_to(var_values, [XBLOCK, RBLOCK])
    broadcast_count = tl.broadcast_to(count_values, [XBLOCK, RBLOCK])

    masked_mean = tl.where(r_mask & x_mask, broadcast_mean, 0)
    masked_var = tl.where(r_mask & x_mask, broadcast_var, 0)
    masked_count = tl.where(r_mask & x_mask, broadcast_count, 0)

    mean_accum, var_accum, count_accum = triton_helpers.welford(masked_mean, masked_var, masked_count, 1)

    reshaped_mean = mean_accum[:, None]
    reshaped_var = var_accum[:, None]

    normalized_var = 640.0
    epsilon = 1e-05
    adjusted_var = reshaped_var / normalized_var
    adjusted_var += epsilon
    inv_sqrt_var = tl.extra.cuda.libdevice.rsqrt(adjusted_var)

    momentum = 0.1
    updated_mean = reshaped_mean * momentum

    decay = 0.9
    decayed_running_mean = running_mean_values * decay
    new_running_mean = updated_mean + decayed_running_mean

    scale_factor = 1.001564945226917
    scaled_var = reshaped_var * scale_factor
    scaled_var *= momentum

    decayed_running_var = running_var_values * decay
    new_running_var = scaled_var + decayed_running_var

    tl.store(output_ptr_normalized + (input_indices), inv_sqrt_var, x_mask)
    tl.store(output_ptr_running_mean + (input_indices), new_running_mean, x_mask)
    tl.store(output_ptr_running_var + (input_indices), new_running_var, x_mask)
    tl.store(output_ptr_mean + (input_indices), reshaped_mean, x_mask)