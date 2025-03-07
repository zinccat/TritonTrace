# From: 79_Conv3d_Multiply_InstanceNorm_Clamp_Multiply_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_clamp_mul_sum_7per_fused_add_clamp_mul_sum_7(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 16
    rnumel = 21
    RBLOCK: tl.constexpr = 32
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_indices = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_indices < rnumel
    r1 = r_indices
    x0 = x_indices
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * r1), r_mask & x_mask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0 + 16 * r1), r_mask & x_mask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(r_mask & x_mask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(r_mask & x_mask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tmp4 + tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp10, x_mask)