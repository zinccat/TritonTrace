# From: 31_VisionAttention

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_6poi_fused_clone_6(input_ptr, output_ptr, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    index_mod_32 = block_indices % 32
    index_div_32_mod_8 = (block_indices // 32) % 8
    index_div_256 = block_indices // 256
    full_index = block_indices
    
    temp_value = tl.load(input_ptr + (index_mod_32 + 32 * index_div_256 + 524288 * index_div_32_mod_8), None)
    tl.store(output_ptr + (full_index), temp_value, None)