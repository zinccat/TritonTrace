# From: 49_ConvTranspose3d_Softmax_Sigmoid

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_per_fused__softmax_1(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 64
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_indices = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = r_indices
    x0 = x_indices % 131072
    x1 = (x_indices // 131072)
    x3 = x_indices
    input_values = tl.load(in_ptr0 + (x0 + (131072 * r2) + (8388608 * x1)), None)
    broadcasted_input = tl.broadcast_to(input_values, [XBLOCK, RBLOCK])
    max_values = triton_helpers.max2(broadcasted_input, 1)[:, None]
    shifted_values = input_values - max_values
    exp_values = tl.math.exp(shifted_values)
    broadcasted_exp = tl.broadcast_to(exp_values, [XBLOCK, RBLOCK])
    sum_exp = tl.sum(broadcasted_exp, 1)[:, None]
    tl.store(out_ptr0 + (x3), max_values, None)
    tl.store(out_ptr1 + (x3), sum_exp, None)