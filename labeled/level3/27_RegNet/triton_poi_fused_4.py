# From: 27_RegNet

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_4poi_fused_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    xnumel = 9
    y_offset = tl.program_id(1) * YBLOCK
    y_index = y_offset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    x_coord = x_index
    y_coord = y_index
    y_modulo = y_index % 128
    y_divide = y_index // 128
    tmp0 = tl.load(in_ptr0 + (x_coord + 9 * y_coord), x_mask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y_modulo + 128 * x_coord + 1152 * y_divide), tmp0, x_mask)