# From: 23_Softmax

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_per_fused__softmax_1(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 16
    RBLOCK: tl.constexpr = 2
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_indices = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = r_indices
    x0 = x_indices
    temp0 = tl.load(in_ptr0 + (r1 + (2 * x0)), x_mask, other=0.0)
    temp1 = tl.broadcast_to(temp0, [XBLOCK, RBLOCK])
    temp3 = tl.where(x_mask, temp1, float("-inf"))
    max_values = triton_helpers.max2(temp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), max_values, x_mask)