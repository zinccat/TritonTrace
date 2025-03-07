# From: 23_EfficientNetB1

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_41(
    input_mean_ptr, input_var_ptr, input_x_ptr, running_mean_ptr, running_var_ptr,
    output_mean_ptr, output_var_ptr, output_x_ptr, output_running_mean_ptr, output_running_var_ptr,
    xnumel, rnumel, XBLOCK: tl.constexpr
):
    xnumel = 80
    rnumel = 18
    RBLOCK: tl.constexpr = 32
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_index = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_index < rnumel
    r1 = r_index
    x0 = x_index

    input_mean = tl.load(input_mean_ptr + (x0 + 80 * r1), r_mask & x_mask, other=0.0)
    input_var = tl.load(input_var_ptr + (x0 + 80 * r1), r_mask & x_mask, other=0.0)
    input_x = tl.load(input_x_ptr + (x0 + 80 * r1), r_mask & x_mask, other=0.0)
    running_mean = tl.load(running_mean_ptr + (x0), x_mask, eviction_policy='evict_last')
    running_var = tl.load(running_var_ptr + (x0), x_mask, eviction_policy='evict_last')

    broadcast_mean = tl.broadcast_to(input_mean, [XBLOCK, RBLOCK])
    broadcast_var = tl.broadcast_to(input_var, [XBLOCK, RBLOCK])
    broadcast_x = tl.broadcast_to(input_x, [XBLOCK, RBLOCK])

    masked_mean = tl.where(r_mask & x_mask, broadcast_mean, 0)
    masked_var = tl.where(r_mask & x_mask, broadcast_var, 0)
    masked_x = tl.where(r_mask & x_mask, broadcast_x, 0)

    mean, var, _ = triton_helpers.welford(masked_mean, masked_var, masked_x, 1)

    mean_reshaped = mean[:, None]
    var_reshaped = var[:, None]

    epsilon = 1e-05
    normalized_var = var_reshaped / 2250.0
    adjusted_var = normalized_var + epsilon
    inv_std = tl.extra.cuda.libdevice.rsqrt(adjusted_var)

    momentum = 0.1
    running_mean_scaled = mean * momentum
    running_mean_updated = running_mean_scaled + running_mean * 0.9

    var_factor = 1.0004446420631392
    normalized_var_scaled = normalized_var * var_factor
    var_scaled = normalized_var_scaled * momentum
    running_var_updated = var_scaled + running_var * 0.9

    tl.store(output_var_ptr + (x0), inv_std, x_mask)
    tl.store(output_running_mean_ptr + (x0), running_mean_updated, x_mask)
    tl.store(output_running_var_ptr + (x0), running_var_updated, x_mask)
    tl.store(output_mean_ptr + (x0), mean_reshaped, x_mask)
    tl.store(output_x_ptr + (x0), var_reshaped, x_mask)