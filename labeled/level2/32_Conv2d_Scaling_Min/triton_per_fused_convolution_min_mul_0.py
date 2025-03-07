# From: 32_Conv2d_Scaling_Min

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_per_fused_convolution_min_mul_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 115200
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    x_indices = xoffset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_indices = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = r_indices
    x0 = x_indices % 900
    x1 = (x_indices // 900)
    x3 = x_indices
    input0 = tl.load(in_ptr0 + (x0 + (900 * r2) + (14400 * x1)), x_mask, other=0.0)
    input1 = tl.load(in_ptr1 + (r2), None, eviction_policy='evict_last')
    sum_inputs = input0 + input1
    scale_factor = 2.0
    scaled_sum = sum_inputs * scale_factor
    broadcasted_scaled_sum = tl.broadcast_to(scaled_sum, [XBLOCK, RBLOCK])
    masked_min = tl.where(x_mask, broadcasted_scaled_sum, float("inf"))
    min_values = triton_helpers.min2(masked_min, 1)[:, None]
    r_index_broadcast = tl.broadcast_to(r_indices, masked_min.shape)
    _, min_indices = triton_helpers.min_with_index(masked_min, r_index_broadcast, 1)
    min_indices = min_indices[:, None]
    tl.store(out_ptr0 + (x3), min_values, x_mask)
    tl.store(out_ptr1 + (x3), min_indices, x_mask)