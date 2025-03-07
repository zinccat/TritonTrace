# From: 36_RMSNorm_

import triton
import triton.language as tl


@triton.jit
def triton_per_fused_mean_pow_0(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 64
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_indices = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = r_indices
    x0 = x_indices % 65536
    x1 = (x_indices // 65536)
    x3 = x_indices
    tmp0 = tl.load(in_ptr0 + (x0 + (65536 * r2) + (4194304 * x1)), None)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp4 = tl.sum(tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, None)