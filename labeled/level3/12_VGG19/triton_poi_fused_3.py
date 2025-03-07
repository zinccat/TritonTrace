# From: 12_VGG19

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_3poi_fused_3(input_ptr, output_ptr, y_num_elements, x_num_elements, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    x_num_elements = 9
    y_offset = tl.program_id(1) * YBLOCK
    y_indices = y_offset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < x_num_elements
    x_coords = x_indices
    y_coords = y_indices
    y_mod_64 = y_indices % 64
    y_div_64 = y_indices // 64
    temp_data = tl.load(input_ptr + (x_coords + 9 * y_coords), x_mask, eviction_policy='evict_last')
    tl.store(output_ptr + (y_mod_64 + 64 * x_coords + 576 * y_div_64), temp_data, x_mask)