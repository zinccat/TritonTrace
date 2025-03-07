# From: 6_Conv3d_Softmax_MaxPool_MaxPool

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__softmax_convolution_0(
    in_ptr0, in_ptr1, out_ptr0, out_ptr1, kernel_size0, kernel_size1, kernel_size2, 
    xnumel, rnumel, XBLOCK: tl.constexpr
):
    RBLOCK: tl.constexpr = 16
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_index = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = r_index
    x3 = (x_index % kernel_size0)
    x4 = x_index // kernel_size0
    x5 = x_index
    tmp0 = tl.load(
        in_ptr0 + (
            x3 + ((-128) * x4) + ((-8) * r2) + 
            ((-32) * x4 * kernel_size2 * kernel_size2) + 
            ((-2) * r2 * kernel_size2 * kernel_size2) + 
            4 * kernel_size1 * r2 + 8 * kernel_size2 * r2 + 
            64 * kernel_size1 * x4 + 128 * kernel_size2 * x4 + 
            kernel_size1 * r2 * kernel_size2 * kernel_size2 + 
            ((-64) * kernel_size1 * kernel_size2 * x4) + 
            ((-4) * kernel_size1 * kernel_size2 * r2) + 
            16 * kernel_size1 * x4 * kernel_size2 * kernel_size2
        ), 
        x_mask, 
        eviction_policy='evict_last', 
        other=0.0
    )
    tmp1 = tl.load(in_ptr1 + (r2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(x_mask, tmp3, float("-inf"))
    tmp6 = triton_helpers.max2(tmp5, 1)[:, None]
    tmp7 = tmp2 - tmp6
    tmp8 = tl.math.exp(tmp7)
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(x_mask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x5), tmp6, x_mask)
    tl.store(out_ptr1 + (x5), tmp12, x_mask)