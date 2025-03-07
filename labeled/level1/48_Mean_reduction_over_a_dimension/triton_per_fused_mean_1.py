# From: 48_Mean_reduction_over_a_dimension

import triton
import triton.language as tl


@triton.jit
def triton_per_fused_mean_1(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 2
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_indices = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = r_indices
    x0 = x_indices % 256
    x1 = (x_indices // 256)
    x3 = x_indices
    tmp0 = tl.load(in_ptr0 + (x0 + (256 * r2) + (512 * x1)), None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.sum(tmp1, 1)[:, None]
    tmp4 = 256.0
    tmp5 = tmp3 / tmp4
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp5, None)