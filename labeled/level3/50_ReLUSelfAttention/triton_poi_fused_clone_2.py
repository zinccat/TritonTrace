# From: 50_ReLUSelfAttention

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_2poi_fused_clone_2(input_ptr0, input_ptr1, output_ptr0, y_num_elements, x_num_elements, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    x_num_elements = 1024
    y_offset = tl.program_id(1) * YBLOCK
    y_indices = y_offset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < x_num_elements
    x_indices_adjusted = x_indices
    y_mod_768 = y_indices % 768
    y_div_768 = y_indices // 768
    y_indices_original = y_indices
    temp0 = tl.load(input_ptr0 + (768 + y_mod_768 + 2304 * x_indices_adjusted + 2359296 * y_div_768), x_mask, eviction_policy='evict_last')
    temp1 = tl.load(input_ptr1 + (768 + y_mod_768), None, eviction_policy='evict_last')
    temp2 = temp0 + temp1
    tl.store(output_ptr0 + (x_indices_adjusted + 1024 * y_indices_original), temp2, x_mask)