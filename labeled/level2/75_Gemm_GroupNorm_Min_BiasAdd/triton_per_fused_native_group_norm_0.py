# From: 75_Gemm_GroupNorm_Min_BiasAdd

import triton
import triton.language as tl


@triton.jit
def triton_per_fused_native_group_norm_0(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 1024
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    x_indices = xoffset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_indices = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = r_indices
    x0 = x_indices
    loaded_values = tl.load(in_ptr0 + (r1 + (32 * x0)), x_mask, other=0.0)
    broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
    tl.where(x_mask, broadcasted_values, 0)
    repeated_values = tl.broadcast_to(broadcasted_values, [XBLOCK, RBLOCK])
    masked_values = tl.where(x_mask, repeated_values, 0)
    sum_across_r = tl.sum(masked_values, 1)[:, None]
    block_size = tl.full([XBLOCK, 1], 32, tl.int32)
    block_size_float = block_size.to(tl.float32)
    mean_values = sum_across_r / block_size_float
    centered_values = broadcasted_values - mean_values
    squared_values = centered_values * centered_values
    broadcasted_squared = tl.broadcast_to(squared_values, [XBLOCK, RBLOCK])
    masked_squared = tl.where(x_mask, broadcasted_squared, 0)
    sum_squared = tl.sum(masked_squared, 1)[:, None]
    variance = sum_squared / 32.0
    epsilon = 1e-05
    variance_with_epsilon = variance + epsilon
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_with_epsilon)
    tl.store(out_ptr2 + (x0), inv_sqrt_variance, x_mask)
    tl.store(out_ptr0 + (x0), mean_values, x_mask)
    tl.store(out_ptr1 + (x0), sum_squared, x_mask)