# From: 53_Min_reduction_over_a_dimension

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_min_0(in_ptr0, out_ptr0, kernel_size0, kernel_size1, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x1 = (x_index // kernel_size0) % 2
    x0 = x_index % kernel_size0
    x2 = x_index // kernel_size1
    _tmp5 = tl.full([XBLOCK, RBLOCK], float("inf"), tl.float32)
    x4 = x_index

    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r3 = r_index
        tmp0 = r3 + x1 * ((1 + kernel_size0) // 2)
        tmp1 = kernel_size0
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(
            in_ptr0 + (x0 + kernel_size0 * r3 + x2 * kernel_size0 * kernel_size0 + kernel_size0 * x1 * ((1 + kernel_size0) // 2)),
            r_mask & tmp2 & x_mask,
            eviction_policy='evict_last',
            other=float("inf")
        )
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = triton_helpers.minimum(_tmp5, tmp4)
        _tmp5 = tl.where(r_mask & x_mask, tmp6, _tmp5)

    tmp5 = triton_helpers.min2(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp5, x_mask)