# From: 39_L2Norm_

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_div_linalg_vector_norm_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    element_indices = block_indices
    block_indices_divided = (block_indices // 16384)
    
    input_value = tl.load(in_ptr0 + (element_indices), None)
    norm_value = tl.load(in_ptr1 + (block_indices_divided), None, eviction_policy='evict_last')
    norm_sqrt = tl.extra.cuda.libdevice.sqrt(norm_value)
    
    result_value = input_value / norm_sqrt
    tl.store(out_ptr0 + (element_indices), result_value, None)