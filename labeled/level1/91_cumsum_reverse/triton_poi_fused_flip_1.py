# From: 91_cumsum_reverse

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_flip_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    index_within_block = block_indices % 4000
    block_number = block_indices // 4000
    original_index = block_indices
    
    reversed_index_within_block = 3999 - index_within_block
    memory_offset = reversed_index_within_block + (4000 * block_number)
    
    temp_value = tl.load(in_ptr0 + memory_offset, None, eviction_policy='evict_last')
    tl.store(out_ptr0 + original_index, temp_value, None)