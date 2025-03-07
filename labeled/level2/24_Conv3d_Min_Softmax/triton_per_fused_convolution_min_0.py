# From: 24_Conv3d_Min_Softmax

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_per_fused_convolution_min_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK: tl.constexpr):
    rnumel = 14
    RBLOCK: tl.constexpr = 16
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_indices = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_indices < rnumel
    r3 = r_indices
    x0 = x_indices % 900
    x4 = (x_indices // 900)
    x1 = (x_indices // 900) % 16
    x5 = x_indices
    tmp0 = tl.load(in_ptr0 + (x0 + (900 * r3) + (12600 * x4)), r_mask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(r_mask, tmp3, float("inf"))
    tmp6 = triton_helpers.min2(tmp5, 1)[:, None]
    tmp8 = tl.broadcast_to(r_indices, tmp5.shape)
    _, tmp7_tmp = triton_helpers.min_with_index(tmp5, tmp8, 1)
    tmp7 = tmp7_tmp[:, None]
    tl.store(out_ptr0 + (x5), tmp6, None)
    tl.store(out_ptr1 + (x5), tmp7, None)