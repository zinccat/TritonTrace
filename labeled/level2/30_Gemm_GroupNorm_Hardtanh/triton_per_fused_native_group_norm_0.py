# From: 30_Gemm_GroupNorm_Hardtanh

import triton
import triton.language as tl


@triton.jit
def triton_per_fused_native_group_norm_0(in_out_ptr0, in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 1024
    RBLOCK: tl.constexpr = 64
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_indices = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = r_indices
    x0 = x_indices
    loaded_values = tl.load(in_ptr0 + (r1 + (64 * x0)), x_mask, other=0.0)
    broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
    tl.where(x_mask, broadcasted_values, 0)
    repeated_broadcasted_values = tl.broadcast_to(broadcasted_values, [XBLOCK, RBLOCK])
    masked_values = tl.where(x_mask, repeated_broadcasted_values, 0)
    sum_across_rows = tl.sum(masked_values, 1)[:, None]
    block_size = tl.full([XBLOCK, 1], 64, tl.int32)
    block_size_float = block_size.to(tl.float32)
    mean_values = sum_across_rows / block_size_float
    centered_values = broadcasted_values - mean_values
    squared_centered_values = centered_values * centered_values
    broadcasted_squared_values = tl.broadcast_to(squared_centered_values, [XBLOCK, RBLOCK])
    masked_squared_values = tl.where(x_mask, broadcasted_squared_values, 0)
    sum_squared_values = tl.sum(masked_squared_values, 1)[:, None]
    normalization_factor = 64.0
    variance_values = sum_squared_values / normalization_factor
    epsilon = 1e-05
    adjusted_variance = variance_values + epsilon
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(adjusted_variance)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), inv_sqrt_variance, x_mask)
    tl.store(out_ptr0 + (x0), mean_values, x_mask)