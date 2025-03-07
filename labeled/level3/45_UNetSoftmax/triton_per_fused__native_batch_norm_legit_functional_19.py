# From: 45_UNetSoftmax

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_19(
    input_mean_ptr, input_var_ptr, output_mean_ptr, output_var_ptr, 
    output_scale_ptr, output_bias_ptr, xnumel, rnumel):

    XBLOCK: tl.constexpr = 1
    RBLOCK: tl.constexpr = 1024

    x_offset = tl.program_id(0) * XBLOCK
    x_index = tl.full([1], x_offset, tl.int32)

    r_index = tl.arange(0, RBLOCK)[:]
    r1 = (r_index % 128)
    r2 = r_index // 128

    x0 = x_index

    input_mean = tl.load(input_mean_ptr + (r1 + 128 * x0 + 131072 * r2), None)
    input_var = tl.load(input_var_ptr + (x0), None, eviction_policy='evict_last')
    output_mean = tl.load(output_mean_ptr + (x0), None, eviction_policy='evict_last')

    broadcast_mean = tl.broadcast_to(input_mean, [RBLOCK])
    sum_mean = triton_helpers.promote_to_tensor(tl.sum(broadcast_mean, 0))
    mean_divisor = tl.full([1], 1024, tl.int32).to(tl.float32)
    normalized_mean = sum_mean / mean_divisor

    mean_diff = broadcast_mean - normalized_mean
    squared_diff = mean_diff * mean_diff
    broadcast_squared_diff = tl.broadcast_to(squared_diff, [RBLOCK])
    sum_squared_diff = triton_helpers.promote_to_tensor(tl.sum(broadcast_squared_diff, 0))
    variance = sum_squared_diff / 1024.0

    epsilon = 1e-05
    adjusted_variance = variance + epsilon
    reciprocal_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(adjusted_variance)

    momentum = 0.1
    moving_mean_update = normalized_mean * momentum
    moving_mean = input_var * 0.9 + moving_mean_update

    variance_scale = 1.0009775171065494
    moving_variance_update = variance * variance_scale * momentum
    moving_variance = input_var * 0.9 + moving_variance_update

    tl.store(output_scale_ptr + (x0), reciprocal_sqrt_variance, None)
    tl.store(output_bias_ptr + (x0), moving_mean, None)
    tl.store(output_var_ptr + (x0), moving_variance, None)
    tl.store(output_mean_ptr + (x0), normalized_mean, None)
    tl.store(input_var_ptr + (x0), variance, None)