# From: 7_GoogleNetInceptionV1

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_9poi_fused_9(input_ptr, output_ptr, total_y_elements, total_x_elements, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    total_y_elements = 25088
    total_x_elements = 9
    y_offset = tl.program_id(1) * YBLOCK
    y_indices = y_offset + tl.arange(0, YBLOCK)[None, :]
    y_mask = y_indices < total_y_elements
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < total_x_elements
    x_indices_adjusted = x_indices
    y_indices_adjusted = y_indices
    y_mod_112 = y_indices % 112
    y_div_112 = y_indices // 112
    temp_data = tl.load(input_ptr + (x_indices_adjusted + 9 * y_indices_adjusted), x_mask & y_mask, eviction_policy='evict_last')
    tl.store(output_ptr + (y_mod_112 + 112 * x_indices_adjusted + 1008 * y_div_112), temp_data, x_mask & y_mask)