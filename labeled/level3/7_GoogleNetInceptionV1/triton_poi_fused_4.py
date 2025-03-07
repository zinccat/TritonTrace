# From: 7_GoogleNetInceptionV1

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_4poi_fused_4(input_ptr, output_ptr, num_elements_y, num_elements_x, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    num_elements_y = 512
    num_elements_x = 25
    y_offset = tl.program_id(1) * YBLOCK
    y_indices = y_offset + tl.arange(0, YBLOCK)[None, :]
    y_mask = y_indices < num_elements_y
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements_x
    x_indices_adjusted = x_indices
    y_indices_adjusted = y_indices
    y_indices_mod = y_indices % 16
    y_indices_div = y_indices // 16
    temp_data = tl.load(input_ptr + (x_indices_adjusted + 25 * y_indices_adjusted), x_mask & y_mask, eviction_policy='evict_last')
    tl.store(output_ptr + (y_indices_mod + 16 * x_indices_adjusted + 400 * y_indices_div), temp_data, x_mask & y_mask)