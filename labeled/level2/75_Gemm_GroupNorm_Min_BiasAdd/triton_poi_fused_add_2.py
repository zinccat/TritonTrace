# From: 75_Gemm_GroupNorm_Min_BiasAdd

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_add_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    index_mod_128 = block_indices % 128
    index_div_128 = block_indices // 128
    original_index = block_indices
    
    input_data0 = tl.load(in_ptr0 + (index_mod_128), None, eviction_policy='evict_last')
    input_data1 = tl.load(in_ptr1 + (index_div_128), None, eviction_policy='evict_last')
    
    result_data = input_data0 + input_data1
    tl.store(out_ptr0 + (original_index), result_data, None)