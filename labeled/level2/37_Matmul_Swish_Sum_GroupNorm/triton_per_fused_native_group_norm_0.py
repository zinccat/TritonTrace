# From: 37_Matmul_Swish_Sum_GroupNorm

import triton
import triton.language as tl


@triton.jit
def triton_per_fused_native_group_norm_0(input_ptr_mean, input_ptr_var, output_ptr_rsqrt, output_ptr_mean, output_ptr_var, xnumel, rnumel, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 32
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_indices = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = r_indices
    x3 = x_indices
    x0 = x_indices % 32
    mean_accumulator = tl.load(input_ptr_mean + (r2 + (32 * x3)), None)
    variance_accumulator = tl.load(input_ptr_var + (r2 + (32 * x0)), None, eviction_policy='evict_last')
    sigmoid_mean = tl.sigmoid(mean_accumulator)
    swish_output = sigmoid_mean * mean_accumulator
    combined_output = swish_output + variance_accumulator
    broadcast_combined = tl.broadcast_to(combined_output, [XBLOCK, RBLOCK])
    sum_broadcast = tl.sum(broadcast_combined, 1)[:, None]
    block_size = tl.full([XBLOCK, 1], 32, tl.int32)
    block_size_float = block_size.to(tl.float32)
    mean = sum_broadcast / block_size_float
    centered_output = broadcast_combined - mean
    squared_centered = centered_output * centered_output
    broadcast_squared = tl.broadcast_to(squared_centered, [XBLOCK, RBLOCK])
    sum_squared = tl.sum(broadcast_squared, 1)[:, None]
    variance = sum_squared / 32.0
    epsilon = 1e-05
    variance_with_epsilon = variance + epsilon
    rsqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_with_epsilon)
    tl.store(output_ptr_rsqrt + (x3), rsqrt_variance, None)
    tl.store(output_ptr_mean + (x3), mean, None)
    tl.store(output_ptr_var + (x3), variance, None)