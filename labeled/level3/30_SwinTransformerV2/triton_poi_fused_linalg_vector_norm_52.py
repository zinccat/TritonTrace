# From: 30_SwinTransformerV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_linalg_vector_norm_52poi_fused_linalg_vector_norm_52(input_ptr, output_ptr, num_elements_y, num_elements_x, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    num_elements_y = 480
    num_elements_x = 49
    y_offset = tl.program_id(1) * YBLOCK
    y_indices = y_offset + tl.arange(0, YBLOCK)[None, :]
    y_mask = y_indices < num_elements_y
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements_x
    x_indices_squared = x_indices
    y0 = (y_indices % 12)
    y1 = y_indices // 12
    y3 = y_indices
    temp0 = tl.load(input_ptr + (y0 + 12 * x_indices_squared + 588 * y1), x_mask & y_mask, eviction_policy='evict_last')
    temp1 = tl.extra.cuda.libdevice.sqrt(temp0)
    tl.store(output_ptr + (x_indices_squared + 49 * y3), temp1, x_mask & y_mask)