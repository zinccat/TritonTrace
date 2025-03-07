# From: 29_SwinMLP

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_38poi_fused_add_38(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK: tl.constexpr):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 384)
    x1 = ((xindex // 384) % 196)
    x2 = xindex // 75264

    # Load data from input pointers
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(
        in_ptr0 + (
            32 * (((4 + (x1 % 14)) % 7)) +
            224 * (((4 + (x1 // 14)) % 7)) +
            1568 * (x0 // 32) +
            18816 * (((4 + (x1 % 14)) // 7)) +
            56448 * (triton_helpers.div_floor_integer(4 + (x1 // 14), 7)) +
            169344 * x2 +
            ((x0 % 32))
        ), 
        xmask
    )
    tmp2 = tl.load(
        in_ptr1 + (
            7 * (((4 + (x1 // 14)) % 7)) +
            49 * (x0 // 32) +
            (((4 + (x1 % 14)) % 7))
        ), 
        xmask, 
        eviction_policy='evict_last'
    )
    tmp5 = tl.load(in_ptr2 + (x3), xmask)
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')

    # Perform computations
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7

    # Store the result back to the output pointer
    tl.store(in_out_ptr0 + (x3), tmp8, xmask)