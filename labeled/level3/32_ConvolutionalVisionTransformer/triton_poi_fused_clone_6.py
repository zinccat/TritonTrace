# From: 32_ConvolutionalVisionTransformer

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_6poi_fused_clone_6(input_ptr, output_ptr, num_elements, BLOCK_SIZE : tl.constexpr):
    num_elements = 2560
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    
    index_mod_128 = block_indices % 128
    index_div_128_mod_2 = (block_indices // 128) % 2
    index_div_256 = block_indices // 256
    original_index = block_indices
    
    temp_value = tl.load(input_ptr + (index_mod_128 + 128 * index_div_256 + 1280 * index_div_128_mod_2), valid_mask)
    tl.store(output_ptr + (original_index), temp_value, valid_mask)