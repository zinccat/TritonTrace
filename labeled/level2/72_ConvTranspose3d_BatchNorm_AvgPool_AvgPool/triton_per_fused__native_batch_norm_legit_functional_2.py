# From: 72_ConvTranspose3d_BatchNorm_AvgPool_AvgPool

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_2(
    input_ptr_mean, input_ptr_var, input_ptr_input, input_ptr_running_mean, input_ptr_running_var,
    output_ptr_normalized, output_ptr_normalized_mean, output_ptr_normalized_var, output_ptr_running_mean, output_ptr_running_var,
    num_elements, num_running_elements, XBLOCK: tl.constexpr
):
    num_elements = 16
    num_running_elements = 249
    RBLOCK: tl.constexpr = 256
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements
    r_indices = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_indices < num_running_elements
    running_indices = r_indices
    input_indices = x_indices

    mean = tl.load(input_ptr_mean + (running_indices + (num_running_elements * input_indices)), r_mask & x_mask, other=0.0)
    var = tl.load(input_ptr_var + (running_indices + (num_running_elements * input_indices)), r_mask & x_mask, other=0.0)
    input_data = tl.load(input_ptr_input + (running_indices + (num_running_elements * input_indices)), r_mask & x_mask, other=0.0)
    running_mean = tl.load(input_ptr_running_mean + (input_indices), x_mask, eviction_policy='evict_last')
    running_var = tl.load(input_ptr_running_var + (input_indices), x_mask, eviction_policy='evict_last')

    mean_broadcast = tl.broadcast_to(mean, [XBLOCK, RBLOCK])
    var_broadcast = tl.broadcast_to(var, [XBLOCK, RBLOCK])
    input_data_broadcast = tl.broadcast_to(input_data, [XBLOCK, RBLOCK])

    mean_masked = tl.where(r_mask & x_mask, mean_broadcast, 0)
    var_masked = tl.where(r_mask & x_mask, var_broadcast, 0)
    input_data_masked = tl.where(r_mask & x_mask, input_data_broadcast, 0)

    mean_accum, var_accum, count = triton_helpers.welford(mean_masked, var_masked, input_data_masked, 1)
    mean_accum_expanded = mean_accum[:, None]
    var_accum_expanded = var_accum[:, None]

    epsilon = 1e-05
    variance_epsilon = var_accum_expanded / 32006016.0 + epsilon
    inv_std = tl.extra.cuda.libdevice.rsqrt(variance_epsilon)

    momentum = 0.1
    adjusted_mean = mean_accum_expanded * momentum
    adjusted_var = var_accum_expanded * momentum

    running_mean_factor = 0.9
    updated_running_mean = running_mean * running_mean_factor + adjusted_mean
    updated_running_var = running_var * running_mean_factor + adjusted_var

    tl.store(output_ptr_normalized + (input_indices), inv_std, x_mask)
    tl.store(output_ptr_running_mean + (input_indices), updated_running_mean, x_mask)
    tl.store(output_ptr_running_var + (input_indices), updated_running_var, x_mask)
    tl.store(output_ptr_normalized_mean + (input_indices), mean_accum_expanded, x_mask)
    tl.store(output_ptr_normalized_var + (input_indices), var_accum_expanded, x_mask)