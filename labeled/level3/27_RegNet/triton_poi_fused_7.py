# From: 27_RegNet

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_7poi_fused_7(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 24
    xnumel = 50176
    y_offset = tl.program_id(1) * YBLOCK
    y_index = y_offset + tl.arange(0, YBLOCK)[None, :]
    y_mask = y_index < ynumel
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    x_coord = x_index
    y_coord = y_index
    y_mod_3 = y_index % 3
    y_div_3 = y_index // 3
    tmp0 = tl.load(in_ptr0 + (x_coord + 50176 * y_coord), x_mask & y_mask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y_mod_3 + 3 * x_coord + 150528 * y_div_3), tmp0, x_mask & y_mask)