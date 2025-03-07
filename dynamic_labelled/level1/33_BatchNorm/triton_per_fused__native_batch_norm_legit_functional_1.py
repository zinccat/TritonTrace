# From: 33_BatchNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_1(
    input_mean_ptr, input_var_ptr, input_x_ptr, running_mean_ptr, running_var_ptr,
    output_mean_ptr, output_var_ptr, output_x_ptr, output_running_mean_ptr, output_running_var_ptr,
    kernel_size_0, kernel_size_1, input_num_elements, running_num_elements, XBLOCK: tl.constexpr
):
    input_num_elements = 64
    running_num_elements = 6
    RBLOCK: tl.constexpr = 8
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < input_num_elements
    r_index = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_index < running_num_elements
    r1 = r_index
    x0 = x_index

    input_mean = tl.load(input_mean_ptr + (x0 + 64 * r1), r_mask & x_mask, other=0.0)
    input_var = tl.load(input_var_ptr + (x0 + 64 * r1), r_mask & x_mask, other=0.0)
    input_x = tl.load(input_x_ptr + (x0 + 64 * r1), r_mask & x_mask, other=0.0)
    running_mean = tl.load(running_mean_ptr + (x0), x_mask, eviction_policy='evict_last')
    running_var = tl.load(running_var_ptr + (x0), x_mask, eviction_policy='evict_last')

    broadcast_mean = tl.broadcast_to(input_mean, [XBLOCK, RBLOCK])
    broadcast_var = tl.broadcast_to(input_var, [XBLOCK, RBLOCK])
    broadcast_x = tl.broadcast_to(input_x, [XBLOCK, RBLOCK])

    masked_mean = tl.where(r_mask & x_mask, broadcast_mean, 0)
    masked_var = tl.where(r_mask & x_mask, broadcast_var, 0)
    masked_x = tl.where(r_mask & x_mask, broadcast_x, 0)

    mean, var, _ = triton_helpers.welford(masked_mean, masked_var, masked_x, 1)

    mean_broadcast = mean[:, None]
    var_broadcast = var[:, None]

    total_elements = kernel_size_0 * kernel_size_1 * kernel_size_1
    total_elements_float = total_elements.to(tl.float32)

    normalized_var = var_broadcast / total_elements_float
    epsilon = 1e-05
    normalized_var_eps = normalized_var + epsilon

    inv_sqrt_var = tl.extra.cuda.libdevice.rsqrt(normalized_var_eps)

    factor = ((64 * kernel_size_0 * kernel_size_1 * kernel_size_1) / 64) / (
        (tl.full([], -1.0, tl.float64)) + ((64 * kernel_size_0 * kernel_size_1 * kernel_size_1) / 64)
    )
    factor_float = factor.to(tl.float32)

    scaled_var = normalized_var * factor_float
    momentum = 0.1
    updated_running_var = scaled_var * momentum

    running_mean_momentum = 0.9
    updated_running_mean = running_mean * running_mean_momentum + updated_running_var

    running_var_momentum = 0.9
    updated_running_var = running_var * running_var_momentum + updated_running_var

    tl.store(output_var_ptr + (x0), inv_sqrt_var, x_mask)
    tl.store(output_running_mean_ptr + (x0), updated_running_mean, x_mask)
    tl.store(output_running_var_ptr + (x0), updated_running_var, x_mask)
    tl.store(output_mean_ptr + (x0), mean_broadcast, x_mask)
    tl.store(output_x_ptr + (x0), var_broadcast, x_mask)