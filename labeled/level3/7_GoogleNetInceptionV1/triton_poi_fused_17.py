# From: 7_GoogleNetInceptionV1

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_17poi_fused_17(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    xnumel = 25
    y_offset = tl.program_id(1) * YBLOCK
    y_index = y_offset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    x_coord = x_index
    y_coord = y_index
    y_modulo = y_index % 48
    y_divide = y_index // 48
    tmp0 = tl.load(in_ptr0 + (x_coord + 25 * y_coord), x_mask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y_modulo + 48 * x_coord + 1200 * y_divide), tmp0, x_mask)