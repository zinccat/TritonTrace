# From: 88_Gemm_GroupNorm_Swish_Multiply_Swish

import triton
import triton.language as tl


@triton.jit
def triton_per_fused_native_group_norm_0(in_out_ptr0, in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 64
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_indices = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = r_indices
    x0 = x_indices
    input_values = tl.load(in_ptr0 + (r1 + (64 * x0)), None)
    broadcasted_input = tl.broadcast_to(input_values, [XBLOCK, RBLOCK])
    tmp3 = tl.broadcast_to(broadcasted_input, [XBLOCK, RBLOCK])
    sum_over_r = tl.sum(tmp3, 1)[:, None]
    block_size = tl.full([XBLOCK, 1], 64, tl.int32)
    block_size_float = block_size.to(tl.float32)
    mean = sum_over_r / block_size_float
    centered_values = broadcasted_input - mean
    squared_values = centered_values * centered_values
    broadcasted_squared = tl.broadcast_to(squared_values, [XBLOCK, RBLOCK])
    sum_of_squares = tl.sum(broadcasted_squared, 1)[:, None]
    variance = sum_of_squares / 64.0
    epsilon = 1e-05
    variance_with_epsilon = variance + epsilon
    inv_stddev = tl.extra.cuda.libdevice.rsqrt(variance_with_epsilon)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), inv_stddev, None)
    tl.store(out_ptr0 + (x0), mean, None)