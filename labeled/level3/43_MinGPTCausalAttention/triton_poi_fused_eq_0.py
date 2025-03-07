# From: 43_MinGPTCausalAttention

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_eq_0poi_fused_eq_0(input_ptr, output_ptr, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    index_mod_512 = block_indices % 512
    index_div_512 = block_indices // 512
    original_index = block_indices
    
    loaded_value = tl.load(input_ptr + (index_mod_512 + 1024 * index_div_512), None)
    comparison_value = 0.0
    is_equal = loaded_value == comparison_value
    
    tl.store(output_ptr + (original_index), is_equal, None)