# From: 11_ConvTranspose2d_BatchNorm_Tanh_MaxPool_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_2(
    input_mean_ptr, input_var_ptr, input_x_ptr, running_mean_ptr, running_var_ptr,
    output_mean_ptr, output_var_ptr, output_x_ptr, output_running_mean_ptr, output_running_var_ptr,
    kernel_size, num_elements, num_running_elements, XBLOCK: tl.constexpr
):
    num_elements = 64
    num_running_elements = 6
    RBLOCK: tl.constexpr = 8
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements
    r_indices = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_indices < num_running_elements
    r1 = r_indices
    x0 = x_indices

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

    mean_expanded = mean[:, None]
    var_expanded = var[:, None]

    normalization_factor = 4096 * kernel_size
    normalization_factor_float = normalization_factor.to(tl.float32)
    epsilon = 1e-05

    inv_std = tl.extra.cuda.libdevice.rsqrt(var_expanded / normalization_factor_float + epsilon)

    scale_factor = (((262144 * kernel_size) / 64) / ((tl.full([], -1.0, tl.float64)) + ((262144 * kernel_size) / 64))).to(tl.float32)
    var_scaled = var_expanded * scale_factor

    momentum = 0.1
    updated_running_mean = running_mean * 0.9 + var_scaled * momentum

    updated_running_var = running_var * 0.9 + var_expanded * momentum

    tl.store(output_mean_ptr + (x0), inv_std, x_mask)
    tl.store(output_var_ptr + (x0), updated_running_mean, x_mask)
    tl.store(output_x_ptr + (x0), updated_running_var, x_mask)
    tl.store(output_running_mean_ptr + (x0), mean_expanded, x_mask)
    tl.store(output_running_var_ptr + (x0), var_expanded, x_mask)