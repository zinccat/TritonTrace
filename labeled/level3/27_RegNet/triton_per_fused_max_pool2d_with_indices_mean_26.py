# From: 27_RegNet

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_max_pool2d_with_indices_mean_26(
    in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr
):
    xnumel = 2048
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_indices = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_indices < rnumel
    r2 = r_indices
    x_mod_256 = x_indices % 256
    x_div_256 = x_indices // 256
    x_full_index = x_indices
    temp0 = tl.load(in_ptr0 + (x_mod_256 + 256 * r2 + 1792 * x_div_256), r_mask & x_mask, other=0.0)
    temp1 = tl.broadcast_to(temp0, [XBLOCK, RBLOCK])
    temp3 = tl.where(r_mask & x_mask, temp1, 0)
    temp4 = tl.sum(temp3, 1)[:, None]
    temp5 = 784.0
    temp6 = temp4 / temp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x_full_index), temp6, x_mask)