# From: 40_LayerNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_native_layer_norm_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr):
    rnumel = 21
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    x_indices = xoffset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_indices = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_indices < rnumel
    row_indices = r_indices
    col_indices = x_indices

    # Load input data with masking
    input_data0 = tl.load(in_ptr0 + (row_indices + 21 * col_indices), r_mask & x_mask, other=0.0)
    input_data1 = tl.load(in_ptr1 + (row_indices + 21 * col_indices), r_mask & x_mask, other=0.0)
    input_data2 = tl.load(in_ptr2 + (row_indices + 21 * col_indices), r_mask & x_mask, other=0.0)

    # Broadcast loaded data
    broadcast_data0 = tl.broadcast_to(input_data0, [XBLOCK, RBLOCK])
    broadcast_data1 = tl.broadcast_to(input_data1, [XBLOCK, RBLOCK])
    broadcast_data2 = tl.broadcast_to(input_data2, [XBLOCK, RBLOCK])

    # Apply mask
    masked_data0 = tl.where(r_mask & x_mask, broadcast_data0, 0)
    masked_data1 = tl.where(r_mask & x_mask, broadcast_data1, 0)
    masked_data2 = tl.where(r_mask & x_mask, broadcast_data2, 0)

    # Compute Welford's algorithm
    mean, variance, count = triton_helpers.welford(masked_data0, masked_data1, masked_data2, 1)

    # Reshape mean and variance
    reshaped_mean = mean[:, None]
    reshaped_variance = variance[:, None]

    # Constants for normalization
    epsilon = 1e-05
    scale_factor = 4194304.0

    # Compute inverse square root of variance
    normalized_variance = reshaped_variance / scale_factor
    adjusted_variance = normalized_variance + epsilon
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(adjusted_variance)

    # Synchronization barrier
    tl.debug_barrier()

    # Store results
    tl.store(in_out_ptr0 + (col_indices), inv_sqrt_variance, x_mask)
    tl.store(out_ptr0 + (col_indices), reshaped_mean, x_mask)