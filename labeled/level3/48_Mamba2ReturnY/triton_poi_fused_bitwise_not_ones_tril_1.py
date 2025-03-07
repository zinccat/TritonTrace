# From: 48_Mamba2ReturnY

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_bitwise_not_ones_tril_1poi_fused_bitwise_not_ones_tril_1(out_ptr0, xnumel, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    index_mod_64 = block_indices % 64
    index_div_64 = block_indices // 64
    original_index = block_indices
    
    tmp0 = index_mod_64 + ((-1) * index_div_64)
    zero_constant = tl.full([1], 0, tl.int64)
    comparison_result = tmp0 <= zero_constant
    true_constant = tl.full([1], True, tl.int1)
    bitwise_and_result = comparison_result & true_constant
    final_result = bitwise_and_result == 0
    
    tl.store(out_ptr0 + (original_index), final_result, None)