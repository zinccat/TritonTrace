# From: 46_NetVladWithGhostClusters

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_linalg_vector_norm_8per_fused_linalg_vector_norm_8(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 32
    RBLOCK: tl.constexpr = 2
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_indices = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = r_indices
    x0 = x_indices
    tmp0 = tl.load(in_ptr0 + (r1 + 2 * x0), x_mask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(x_mask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = tl.extra.cuda.libdevice.sqrt(tmp4)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp5, x_mask)