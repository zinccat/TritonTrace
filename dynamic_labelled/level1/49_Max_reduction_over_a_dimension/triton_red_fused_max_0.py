# From: 49_Max_reduction_over_a_dimension

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_max_0red_fused_max_0(in_ptr0, out_ptr0, kernel_size0, kernel_size1, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // kernel_size0) % 2
    x0 = xindex % kernel_size0
    x2 = xindex // kernel_size1
    _max_values = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    x4 = xindex

    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + x1 * ((1 + kernel_size0) // 2)
        tmp1 = kernel_size0
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + kernel_size0 * r3 + x2 * kernel_size0 * kernel_size0 + kernel_size0 * x1 * ((1 + kernel_size0) // 2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=float("-inf"))
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = triton_helpers.maximum(_max_values, tmp4)
        _max_values = tl.where(rmask & xmask, tmp6, _max_values)

    max_result = triton_helpers.max2(_max_values, 1)[:, None]
    tl.store(out_ptr0 + (x4), max_result, xmask)