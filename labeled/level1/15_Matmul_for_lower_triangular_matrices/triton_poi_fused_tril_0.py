# From: 15_Matmul_for_lower_triangular_matrices

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_tril_0(in_out_ptr0, xnumel, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    index_mod_4096 = block_indices % 4096
    index_div_4096 = block_indices // 4096
    original_index = block_indices
    
    loaded_value = tl.load(in_out_ptr0 + original_index, None)
    adjusted_index = index_mod_4096 + ((-1) * index_div_4096)
    zero_value = tl.full([1], 0, tl.int64)
    condition = adjusted_index <= zero_value
    
    zero_float = 0.0
    result_value = tl.where(condition, loaded_value, zero_float)
    
    tl.store(in_out_ptr0 + original_index, result_value, None)