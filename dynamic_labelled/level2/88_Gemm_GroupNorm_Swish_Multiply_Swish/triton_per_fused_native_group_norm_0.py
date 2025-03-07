# From: 88_Gemm_GroupNorm_Swish_Multiply_Swish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_native_group_norm_0per_fused_native_group_norm_0(in_out_ptr0, in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    input_values = tl.load(in_ptr0 + (r1 + 64 * x0), xmask, other=0.0)
    broadcasted_input = tl.broadcast_to(input_values, [XBLOCK, RBLOCK])
    tl.where(xmask, broadcasted_input, 0)
    repeated_broadcast = tl.broadcast_to(broadcasted_input, [XBLOCK, RBLOCK])
    masked_broadcast = tl.where(xmask, repeated_broadcast, 0)
    sum_over_r = tl.sum(masked_broadcast, 1)[:, None]
    num_elements = tl.full([XBLOCK, 1], 64, tl.int32)
    num_elements_float = num_elements.to(tl.float32)
    mean = sum_over_r / num_elements_float
    centered_values = broadcasted_input - mean
    squared_centered = centered_values * centered_values
    broadcasted_squared = tl.broadcast_to(squared_centered, [XBLOCK, RBLOCK])
    masked_squared = tl.where(xmask, broadcasted_squared, 0)
    sum_squared = tl.sum(masked_squared, 1)[:, None]
    variance = sum_squared / 64.0
    epsilon = 1e-05
    variance_epsilon = variance + epsilon
    inv_stddev = tl.extra.cuda.libdevice.rsqrt(variance_epsilon)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), inv_stddev, xmask)
    tl.store(out_ptr0 + (x0), mean, xmask)