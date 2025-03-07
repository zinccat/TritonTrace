# From: 34_VanillaRNNHidden

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_0poi_fused_cat_0(input_ptr, output_ptr, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    index_within_block = block_indices
    index_modulo = block_indices % 1024
    index_divide = block_indices // 1024
    
    temp_data = tl.load(input_ptr + (index_within_block), None)
    tl.store(output_ptr + (index_modulo + 1280 * index_divide), temp_data, None)