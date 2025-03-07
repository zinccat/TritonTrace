# From: 48_Mamba2ReturnY

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_11poi_fused_clone_11(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    xnumel = 64
    y_offset = tl.program_id(1) * YBLOCK
    y_index = y_offset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    x4 = x_index
    y0 = (y_index % 16)
    y1 = ((y_index // 16) % 8)
    y2 = ((y_index // 128) % 2)
    y3 = y_index // 256
    y5 = y_index
    tmp0 = tl.load(in_ptr0 + (y0 + 16*x4 + 1024*y2 + 3072*y1 + 24576*y3), x_mask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x4 + 64*y5), tmp0, x_mask)