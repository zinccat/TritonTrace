# From: 40_LayerNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_native_layer_norm_1per_fused_native_layer_norm_1(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr
):
    rnumel = 21
    RBLOCK: tl.constexpr = 32
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_indices = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_indices < rnumel
    r1 = r_indices
    x0 = x_indices

    mean_values = tl.load(in_ptr0 + (r1 + 21 * x0), r_mask & x_mask, other=0.0)
    variance_values = tl.load(in_ptr1 + (r1 + 21 * x0), r_mask & x_mask, other=0.0)
    count_values = tl.load(in_ptr2 + (r1 + 21 * x0), r_mask & x_mask, other=0.0)

    broadcast_mean = tl.broadcast_to(mean_values, [XBLOCK, RBLOCK])
    broadcast_variance = tl.broadcast_to(variance_values, [XBLOCK, RBLOCK])
    broadcast_count = tl.broadcast_to(count_values, [XBLOCK, RBLOCK])

    masked_mean = tl.where(r_mask & x_mask, broadcast_mean, 0)
    masked_variance = tl.where(r_mask & x_mask, broadcast_variance, 0)
    masked_count = tl.where(r_mask & x_mask, broadcast_count, 0)

    mean, variance, count = triton_helpers.welford(masked_mean, masked_variance, masked_count, 1)

    reshaped_mean = mean[:, None]
    reshaped_variance = variance[:, None]

    epsilon = 1e-05
    normalized_variance = reshaped_variance / 4194304.0
    adjusted_variance = normalized_variance + epsilon
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(adjusted_variance)

    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), inv_sqrt_variance, x_mask)
    tl.store(out_ptr0 + (x0), reshaped_mean, x_mask)