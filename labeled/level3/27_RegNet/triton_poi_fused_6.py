# From: 27_RegNet

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_6poi_fused_6(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 65536
    xnumel = 9
    y_offset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    y_index = y_offset + tl.arange(0, YBLOCK)[None, :]
    y_mask = y_index < ynumel
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    x_coord = x_index
    y_coord = y_index
    y_mod = y_index % 256
    y_div = y_index // 256
    tmp_data = tl.load(in_ptr0 + (x_coord + 9 * y_coord), x_mask & y_mask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y_mod + 256 * x_coord + 2304 * y_div), tmp_data, x_mask & y_mask)