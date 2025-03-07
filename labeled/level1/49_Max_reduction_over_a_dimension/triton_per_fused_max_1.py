# From: 49_Max_reduction_over_a_dimension

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_per_fused_max_1(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 2
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_indices = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = r_indices
    x_mod_256 = x_indices % 256
    x_div_256 = (x_indices // 256)
    x_full_indices = x_indices
    temp0 = tl.load(in_ptr0 + (x_mod_256 + (256 * r2) + (512 * x_div_256)), None)
    temp1 = tl.broadcast_to(temp0, [XBLOCK, RBLOCK])
    temp3 = triton_helpers.max2(temp1, 1)[:, None]
    tl.store(out_ptr0 + (x_full_indices), temp3, None)