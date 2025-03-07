# From: 38_ConvTranspose3d_AvgPool_Clamp_Softmax_Multiply

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_per_fused__softmax_clamp_2(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 16
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_indices = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = r_indices
    x0 = x_indices % 16384
    x1 = (x_indices // 16384)
    x3 = x_indices
    input_value = tl.load(in_ptr0 + (x0 + (16384 * r2) + (262144 * x1)), None)
    clamp_min = 0.0
    clamped_value = triton_helpers.maximum(input_value, clamp_min)
    clamp_max = 1.0
    clamped_value = triton_helpers.minimum(clamped_value, clamp_max)
    broadcast_clamped = tl.broadcast_to(clamped_value, [XBLOCK, RBLOCK])
    max_clamped = triton_helpers.max2(broadcast_clamped, 1)[:, None]
    shifted_values = clamped_value - max_clamped
    exp_values = tl.math.exp(shifted_values)
    broadcast_exp = tl.broadcast_to(exp_values, [XBLOCK, RBLOCK])
    sum_exp = tl.sum(broadcast_exp, 1)[:, None]
    tl.store(out_ptr0 + (x3), max_clamped, None)
    tl.store(out_ptr1 + (x3), sum_exp, None)