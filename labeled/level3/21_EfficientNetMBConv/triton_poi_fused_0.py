# From: 21_EfficientNetMBConv

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_0poi_fused_0(input_ptr, output_ptr, num_elements_y, num_elements_x, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    num_elements_y = 1120
    num_elements_x = 50176
    y_offset = tl.program_id(1) * YBLOCK
    y_indices = y_offset + tl.arange(0, YBLOCK)[None, :]
    y_mask = y_indices < num_elements_y
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements_x
    x_indices_2d = x_indices
    y_indices_2d = y_indices
    y_mod = y_indices % 112
    y_div = y_indices // 112
    temp_data = tl.load(input_ptr + (x_indices_2d + 50176 * y_indices_2d), x_mask & y_mask, eviction_policy='evict_last')
    tl.store(output_ptr + (y_mod + 112 * x_indices_2d + 5619712 * y_div), temp_data, x_mask & y_mask)