# From: 7_GoogleNetInceptionV1

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_13poi_fused_13(input_ptr, output_ptr, y_num_elements, x_num_elements, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    x_num_elements = 25
    y_offset = tl.program_id(1) * YBLOCK
    y_indices = y_offset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < x_num_elements
    x_indices_adjusted = x_indices
    y_indices_adjusted = y_indices
    y_modulo_32 = y_indices % 32
    y_div_32 = y_indices // 32
    temp_data = tl.load(input_ptr + (x_indices_adjusted + 25 * y_indices_adjusted), x_mask, eviction_policy='evict_last')
    tl.store(output_ptr + (y_modulo_32 + 32 * x_indices_adjusted + 800 * y_div_32), temp_data, x_mask)