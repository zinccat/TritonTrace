# From: 12_Matmul_with_diagonal_matrices_

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_diag_embed_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    index_mod = block_indices % 4096
    index_div = block_indices // 4096
    original_index = block_indices
    
    input_value = tl.load(in_ptr0 + (index_mod), None, eviction_policy='evict_last')
    mod_index = index_mod
    div_index = index_div
    is_diagonal = mod_index == div_index
    
    zero_value = 0.0
    output_value = tl.where(is_diagonal, input_value, zero_value)
    
    tl.store(out_ptr0 + (original_index), output_value, None)