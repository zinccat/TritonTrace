# From: 33_BatchNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_native_batch_norm_backward_2(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 64
    rnumel = 6
    RBLOCK: tl.constexpr = 8
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_indices = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_indices < rnumel
    r1 = r_indices
    x0 = x_indices
    tmp0 = tl.load(in_ptr0 + (x0 + 64 * r1), r_mask & x_mask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), x_mask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(r_mask & x_mask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp6, x_mask)