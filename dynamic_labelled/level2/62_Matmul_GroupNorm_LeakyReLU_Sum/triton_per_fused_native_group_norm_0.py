# From: 62_Matmul_GroupNorm_LeakyReLU_Sum

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_native_group_norm_0per_fused_native_group_norm_0(in_out_ptr0, in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 32
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_indices = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = r_indices
    x0 = x_indices
    input_values = tl.load(in_ptr0 + (r1 + 32 * x0), x_mask, other=0.0)
    broadcasted_input = tl.broadcast_to(input_values, [XBLOCK, RBLOCK])
    tl.where(x_mask, broadcasted_input, 0)
    expanded_input = tl.broadcast_to(broadcasted_input, [XBLOCK, RBLOCK])
    masked_input = tl.where(x_mask, expanded_input, 0)
    sum_across_r = tl.sum(masked_input, 1)[:, None]
    block_size = tl.full([XBLOCK, 1], 32, tl.int32)
    block_size_float = block_size.to(tl.float32)
    mean = sum_across_r / block_size_float
    centered_values = broadcasted_input - mean
    squared_values = centered_values * centered_values
    expanded_squared = tl.broadcast_to(squared_values, [XBLOCK, RBLOCK])
    masked_squared = tl.where(x_mask, expanded_squared, 0)
    sum_squared = tl.sum(masked_squared, 1)[:, None]
    variance = sum_squared / 32.0
    epsilon = 1e-05
    variance_epsilon = variance + epsilon
    inv_std = tl.extra.cuda.libdevice.rsqrt(variance_epsilon)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), inv_std, x_mask)
    tl.store(out_ptr0 + (x0), mean, x_mask)