# From: 38_L1Norm_

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_div_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    index_within_block = block_indices
    index_for_in_ptr1 = block_indices // 16384
    
    input_value0 = tl.load(in_ptr0 + index_within_block, None)
    input_value1 = tl.load(in_ptr1 + index_for_in_ptr1, None, eviction_policy='evict_last')
    
    result = input_value0 / input_value1
    tl.store(out_ptr0 + index_within_block, result, None)