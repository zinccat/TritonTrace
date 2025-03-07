# From: 37_Matmul_Swish_Sum_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_native_group_norm_0(
    input_ptr_mean, input_ptr_var, output_ptr_rsqrt, output_ptr_mean, output_ptr_var, 
    xnumel, rnumel, XBLOCK: tl.constexpr
):
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_block_index = rindex
    x_block_index = xindex
    x_within_block_index = xindex % 32
    mean_accumulator = tl.load(input_ptr_mean + (r_block_index + 32 * x_block_index), xmask, other=0.0)
    var_accumulator = tl.load(input_ptr_var + (r_block_index + 32 * x_within_block_index), xmask, eviction_policy='evict_last', other=0.0)
    sigmoid_mean = tl.sigmoid(mean_accumulator)
    swish_result = sigmoid_mean * mean_accumulator
    combined_result = swish_result + var_accumulator
    broadcasted_combined = tl.broadcast_to(combined_result, [XBLOCK, RBLOCK])
    tl.where(xmask, broadcasted_combined, 0)
    broadcasted_combined_2 = tl.broadcast_to(combined_result, [XBLOCK, RBLOCK])
    masked_combined = tl.where(xmask, broadcasted_combined_2, 0)
    mean_sum = tl.sum(masked_combined, 1)[:, None]
    block_size = tl.full([XBLOCK, 1], 32, tl.int32)
    block_size_float = block_size.to(tl.float32)
    mean = mean_sum / block_size_float
    normalized_result = combined_result - mean
    squared_normalized = normalized_result * normalized_result
    broadcasted_squared = tl.broadcast_to(squared_normalized, [XBLOCK, RBLOCK])
    masked_squared = tl.where(xmask, broadcasted_squared, 0)
    variance_sum = tl.sum(masked_squared, 1)[:, None]
    variance = variance_sum / 32.0
    epsilon = 1e-05
    variance_with_epsilon = variance + epsilon
    rsqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_with_epsilon)
    tl.store(output_ptr_rsqrt + (x_block_index), rsqrt_variance, xmask)
    tl.store(output_ptr_mean + (x_block_index), mean, xmask)
    tl.store(output_ptr_var + (x_block_index), variance_sum, xmask)