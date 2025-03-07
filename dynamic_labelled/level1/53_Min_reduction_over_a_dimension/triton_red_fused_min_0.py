# From: 53_Min_reduction_over_a_dimension

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_min_0red_fused_min_0(in_ptr0, out_ptr0, kernel_size0, kernel_size1, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    
    x1 = (xindex // kernel_size0) % 2
    x0 = xindex % kernel_size0
    x2 = xindex // kernel_size1
    
    min_values = tl.full([XBLOCK, RBLOCK], float("inf"), tl.float32)
    x4 = xindex
    
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        
        tmp0 = r3 + x1 * ((1 + kernel_size0) // 2)
        tmp1 = kernel_size0
        tmp2 = tmp0 < tmp1
        
        loaded_values = tl.load(
            in_ptr0 + (x0 + kernel_size0 * r3 + x2 * kernel_size0 * kernel_size0 + kernel_size0 * x1 * ((1 + kernel_size0) // 2)),
            rmask & tmp2 & xmask,
            eviction_policy='evict_last',
            other=float("inf")
        )
        
        broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
        min_values = triton_helpers.minimum(min_values, broadcasted_values)
        min_values = tl.where(rmask & xmask, min_values, min_values)
    
    min_result = triton_helpers.min2(min_values, 1)[:, None]
    tl.store(out_ptr0 + (x4), min_result, xmask)