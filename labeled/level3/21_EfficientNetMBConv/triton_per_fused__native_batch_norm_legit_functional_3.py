# From: 21_EfficientNetMBConv

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_3(
    input_ptr_mean, input_ptr_var, input_ptr_count, input_ptr_running_mean, input_ptr_running_var,
    output_ptr_normalized, output_ptr_running_mean, output_ptr_running_var, output_ptr_mean,
    num_elements, running_elements, XBLOCK: tl.constexpr
):
    num_elements = 672
    RBLOCK: tl.constexpr = 2
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements
    r_indices = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = r_indices
    x0 = x_indices

    mean = tl.load(input_ptr_mean + (x0 + 672 * r1), x_mask, other=0.0)
    variance = tl.load(input_ptr_var + (x0 + 672 * r1), x_mask, other=0.0)
    count = tl.load(input_ptr_count + (x0 + 672 * r1), x_mask, other=0.0)
    running_mean = tl.load(input_ptr_running_mean + (x0), x_mask, eviction_policy='evict_last')
    running_var = tl.load(input_ptr_running_var + (x0), x_mask, eviction_policy='evict_last')

    mean_broadcast = tl.broadcast_to(mean, [XBLOCK, RBLOCK])
    variance_broadcast = tl.broadcast_to(variance, [XBLOCK, RBLOCK])
    count_broadcast = tl.broadcast_to(count, [XBLOCK, RBLOCK])

    masked_mean = tl.where(x_mask, mean_broadcast, 0)
    masked_variance = tl.where(x_mask, variance_broadcast, 0)
    masked_count = tl.where(x_mask, count_broadcast, 0)

    mean_accum, variance_accum, count_accum = triton_helpers.welford(
        masked_mean, masked_variance, masked_count, 1
    )

    mean_accum_expanded = mean_accum[:, None]
    variance_accum_expanded = variance_accum[:, None]

    num_running_elements = 501760.0
    variance_normalized = variance_accum_expanded / num_running_elements
    epsilon = 1e-05
    variance_normalized_eps = variance_normalized + epsilon
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_normalized_eps)

    variance_scale = 1.0000019929886659
    variance_scaled = variance_normalized * variance_scale
    momentum = 0.1
    variance_momentum = variance_scaled * momentum
    running_mean_scale = 0.9
    updated_running_mean = running_mean * running_mean_scale
    running_mean_momentum = mean_accum_expanded * momentum
    new_running_mean = variance_momentum + updated_running_mean

    updated_running_var = running_var * running_mean_scale
    new_running_var = running_mean_momentum + updated_running_var

    tl.store(output_ptr_normalized + (x0), inv_sqrt_variance, x_mask)
    tl.store(output_ptr_running_mean + (x0), new_running_mean, x_mask)
    tl.store(output_ptr_running_var + (x0), new_running_var, x_mask)
    tl.store(output_ptr_mean + (x0), mean_accum_expanded, x_mask)