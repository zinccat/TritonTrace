# From: 48_Mamba2ReturnY

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_3poi_fused_clone_3(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    xnumel = 64
    y_offset = tl.program_id(1) * YBLOCK
    y_index = y_offset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    x2 = x_index
    y0 = (y_index % 128)
    y1 = y_index // 128
    y7 = y_index
    y4 = ((y_index // 16) % 8)
    y5 = ((y_index // 128) % 2)
    y6 = y_index // 256
    tmp0 = tl.load(in_ptr0 + (y0 + 128 * x2 + 8192 * y1), x_mask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (63 + 64 * y5 + 128 * y4 + 1024 * y6), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x2 + 64 * y5 + 128 * y4 + 1024 * y6), x_mask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp4 = tl.math.exp(tmp3)
    tmp5 = tmp0 * tmp4
    tl.store(out_ptr0 + (x2 + 64 * y7), tmp0, x_mask)
    tl.store(out_ptr1 + (x2 + 64 * y7), tmp5, x_mask)