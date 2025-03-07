# From: 43_MinGPTCausalAttention

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_2poi_fused_clone_2(input_ptr0, input_ptr1, output_ptr0, y_num_elements, x_num_elements, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    y_num_elements = 98304
    x_num_elements = 512
    
    y_offset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    y_index = y_offset + tl.arange(0, YBLOCK)[None, :]
    y_mask = y_index < y_num_elements
    
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < x_num_elements
    
    x_index_adjusted = x_index
    y_index_mod = y_index % 768
    y_index_div = y_index // 768
    y_index_original = y_index
    
    temp0 = tl.load(input_ptr0 + (768 + y_index_mod + 2304 * x_index_adjusted + 1179648 * y_index_div), x_mask & y_mask, eviction_policy='evict_last')
    temp1 = tl.load(input_ptr1 + (768 + y_index_mod), y_mask, eviction_policy='evict_last')
    temp2 = temp0 + temp1
    
    tl.store(output_ptr0 + (x_index_adjusted + 512 * y_index_original), temp2, x_mask & y_mask)