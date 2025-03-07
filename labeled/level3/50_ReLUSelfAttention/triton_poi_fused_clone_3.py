# From: 50_ReLUSelfAttention

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_3poi_fused_clone_3(input_ptr0, input_ptr1, output_ptr0, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    index_mod_64 = block_indices % 64
    index_div_64_mod_1024 = (block_indices // 64) % 1024
    index_div_65536_mod_12 = (block_indices // 65536) % 12
    index_div_786432 = block_indices // 786432
    full_index = block_indices
    
    tmp0 = tl.load(input_ptr0 + (1536 + index_mod_64 + 64 * index_div_65536_mod_12 + 2304 * index_div_64_mod_1024 + 2359296 * index_div_786432), None)
    tmp1 = tl.load(input_ptr1 + (1536 + index_mod_64 + 64 * index_div_65536_mod_12), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(output_ptr0 + (full_index), tmp2, None)