# From: 11_VGG16

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_6poi_fused_6(input_ptr, output_ptr, total_y_elements, total_x_elements, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    total_y_elements = 65536
    total_x_elements = 9
    y_offset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    y_index = y_offset + tl.arange(0, YBLOCK)[None, :]
    y_mask = y_index < total_y_elements
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < total_x_elements
    x_coord = x_index
    y_coord = y_index
    y_mod = y_index % 256
    y_div = y_index // 256
    temp_data = tl.load(input_ptr + (x_coord + 9 * y_coord), x_mask & y_mask, eviction_policy='evict_last')
    tl.store(output_ptr + (y_mod + 256 * x_coord + 2304 * y_div), temp_data, x_mask & y_mask)