# From: 50_Product_reduction_over_a_dimension

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_prod_1(in_ptr0, out_ptr0, ks0, xnumel, rnumel, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 2
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_indices = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = r_indices
    x_mod_ks0 = x_indices % ks0
    x_div_ks0 = x_indices // ks0
    x_full_indices = x_indices
    tmp0 = tl.load(in_ptr0 + (x_mod_ks0 + ks0 * r2 + 2 * ks0 * x_div_ks0), x_mask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(x_mask, tmp1, 1)
    tmp4 = triton_helpers.prod(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x_full_indices), tmp4, x_mask)