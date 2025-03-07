# From: 37_Matmul_Swish_Sum_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_native_group_norm_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    row_indices = rindex
    col_indices = xindex
    col_modulo = xindex % 32
    input_data = tl.load(in_ptr0 + (row_indices + 32 * col_indices), xmask, other=0.0)
    bias_data = tl.load(in_ptr1 + (row_indices + 32 * col_modulo), xmask, eviction_policy='evict_last', other=0.0)
    sigmoid_output = tl.sigmoid(input_data)
    elementwise_product = sigmoid_output * input_data
    sum_with_bias = elementwise_product + bias_data
    broadcast_sum = tl.broadcast_to(sum_with_bias, [XBLOCK, RBLOCK])
    tl.where(xmask, broadcast_sum, 0)
    broadcast_sum_repeated = tl.broadcast_to(broadcast_sum, [XBLOCK, RBLOCK])
    masked_broadcast_sum = tl.where(xmask, broadcast_sum_repeated, 0)
    sum_over_rows = tl.sum(masked_broadcast_sum, 1)[:, None]
    block_size = tl.full([XBLOCK, 1], 32, tl.int32)
    block_size_float = block_size.to(tl.float32)
    mean = sum_over_rows / block_size_float
    normalized_data = broadcast_sum - mean
    squared_normalized_data = normalized_data * normalized_data
    broadcast_squared = tl.broadcast_to(squared_normalized_data, [XBLOCK, RBLOCK])
    masked_squared = tl.where(xmask, broadcast_squared, 0)
    sum_of_squares = tl.sum(masked_squared, 1)[:, None]
    variance = sum_of_squares / 32.0
    epsilon = 1e-05
    variance_with_epsilon = variance + epsilon
    reciprocal_sqrt = tl.extra.cuda.libdevice.rsqrt(variance_with_epsilon)
    tl.store(out_ptr2 + (col_indices), reciprocal_sqrt, xmask)
    tl.store(out_ptr0 + (col_indices), mean, xmask)
    tl.store(out_ptr1 + (col_indices), sum_of_squares, xmask)