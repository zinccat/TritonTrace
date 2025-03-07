# From: 14_Matmul_for_upper_triangular_matrices

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_triu_0(in_out_ptr0, xnumel, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    index_mod = block_indices % 4096
    index_div = block_indices // 4096
    linear_index = block_indices
    
    loaded_value = tl.load(in_out_ptr0 + (linear_index), None)
    adjusted_index = index_mod + ((-1) * index_div)
    zero_value = tl.full([1], 0, tl.int64)
    is_valid_index = adjusted_index >= zero_value
    
    zero_float = 0.0
    result_value = tl.where(is_valid_index, loaded_value, zero_float)
    
    tl.store(in_out_ptr0 + (linear_index), result_value, None)