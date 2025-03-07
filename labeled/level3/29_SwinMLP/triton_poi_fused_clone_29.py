# From: 29_SwinMLP

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_29poi_fused_clone_29(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x_mod_32 = xindex % 32
    x_div_32_mod_49 = (xindex // 32) % 49
    x_div_1568_mod_12 = (xindex // 1568) % 12
    x_div_18816 = xindex // 18816
    x_full_index = xindex

    tmp0 = tl.load(
        in_ptr0 + (x_mod_32 + 32 * x_div_1568_mod_12 + 384 * (x_div_32_mod_49 % 7) + 2688 * (x_div_18816 % 2) + 5376 * (x_div_32_mod_49 // 7) + 37632 * (x_div_18816 // 2)),
        xmask
    )
    tmp1 = tl.load(
        in_ptr1 + (7 * (x_div_18816 % 2) + 14 * (x_div_32_mod_49 // 7) + 98 * (x_div_18816 // 2) + (x_div_32_mod_49 % 7)),
        xmask,
        eviction_policy='evict_last'
    )
    tmp3 = tl.load(
        in_ptr2 + (7 * (x_div_18816 % 2) + 14 * (x_div_32_mod_49 // 7) + 98 * (x_div_18816 // 2) + (x_div_32_mod_49 % 7)),
        xmask,
        eviction_policy='evict_last'
    )
    tmp5 = tl.load(
        in_ptr3 + (x_mod_32 + 32 * x_div_1568_mod_12),
        xmask,
        eviction_policy='evict_last'
    )
    tmp7 = tl.load(
        in_ptr4 + (x_mod_32 + 32 * x_div_1568_mod_12),
        xmask,
        eviction_policy='evict_last'
    )

    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7

    tl.store(out_ptr0 + (x_full_index), tmp8, xmask)