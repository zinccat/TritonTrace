# From: 75_Gemm_GroupNorm_Min_BiasAdd

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_native_group_norm_0per_fused_native_group_norm_0(
    input_ptr, output_mean_ptr, output_var_ptr, output_rsqrt_ptr, 
    xnumel, rnumel, XBLOCK: tl.constexpr
):
    RBLOCK: tl.constexpr = 32
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_indices = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_indices_broadcast = r_indices
    x_indices_broadcast = x_indices
    input_values = tl.load(input_ptr + (r_indices_broadcast + 32 * x_indices_broadcast), x_mask, other=0.0)
    input_broadcast = tl.broadcast_to(input_values, [XBLOCK, RBLOCK])
    tl.where(x_mask, input_broadcast, 0)
    input_broadcast_2 = tl.broadcast_to(input_broadcast, [XBLOCK, RBLOCK])
    masked_input = tl.where(x_mask, input_broadcast_2, 0)
    sum_input = tl.sum(masked_input, 1)[:, None]
    block_size = tl.full([XBLOCK, 1], 32, tl.int32)
    block_size_float = block_size.to(tl.float32)
    mean = sum_input / block_size_float
    centered_input = input_broadcast - mean
    squared_centered_input = centered_input * centered_input
    squared_centered_broadcast = tl.broadcast_to(squared_centered_input, [XBLOCK, RBLOCK])
    masked_squared_centered = tl.where(x_mask, squared_centered_broadcast, 0)
    sum_squared_centered = tl.sum(masked_squared_centered, 1)[:, None]
    variance = sum_squared_centered / 32.0
    epsilon = 1e-05
    variance_with_epsilon = variance + epsilon
    rsqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_with_epsilon)
    tl.store(output_rsqrt_ptr + (x_indices_broadcast), rsqrt_variance, x_mask)
    tl.store(output_mean_ptr + (x_indices_broadcast), mean, x_mask)
    tl.store(output_var_ptr + (x_indices_broadcast), sum_squared_centered, x_mask)