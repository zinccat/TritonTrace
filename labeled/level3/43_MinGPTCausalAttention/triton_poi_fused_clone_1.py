# From: 43_MinGPTCausalAttention

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_1poi_fused_clone_1(input_ptr0, input_ptr1, output_ptr0, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    index_mod_96 = block_indices % 96
    index_div_96_mod_512 = (block_indices // 96) % 512
    index_div_49152_mod_8 = (block_indices // 49152) % 8
    index_div_393216 = block_indices // 393216
    linear_index = block_indices
    
    tmp0 = tl.load(input_ptr0 + (index_mod_96 + 96 * index_div_49152_mod_8 + 2304 * index_div_96_mod_512 + 1179648 * index_div_393216), None)
    tmp1 = tl.load(input_ptr1 + (index_mod_96 + 96 * index_div_49152_mod_8), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(output_ptr0 + (linear_index), tmp2, None)