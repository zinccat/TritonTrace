# From: 24_Conv3d_Min_Softmax

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_per_fused__softmax_1(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 115200
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    x_indices = xoffset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_indices = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = r_indices
    x_mod_900 = x_indices % 900
    x_div_900 = (x_indices // 900)
    x_full_index = x_indices
    input_values = tl.load(in_ptr0 + (x_mod_900 + (900 * r2) + (14400 * x_div_900)), x_mask, other=0.0)
    broadcasted_values = tl.broadcast_to(input_values, [XBLOCK, RBLOCK])
    masked_values = tl.where(x_mask, broadcasted_values, float("-inf"))
    max_values = triton_helpers.max2(masked_values, 1)[:, None]
    shifted_values = input_values - max_values
    exp_values = tl.math.exp(shifted_values)
    broadcasted_exp_values = tl.broadcast_to(exp_values, [XBLOCK, RBLOCK])
    masked_exp_values = tl.where(x_mask, broadcasted_exp_values, 0)
    sum_exp_values = tl.sum(masked_exp_values, 1)[:, None]
    tl.store(out_ptr0 + (x_full_index), max_values, x_mask)
    tl.store(out_ptr1 + (x_full_index), sum_exp_values, x_mask)