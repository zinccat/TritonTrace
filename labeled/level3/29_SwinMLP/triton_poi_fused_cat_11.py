# From: 29_SwinMLP

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_11poi_fused_cat_11(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 752640
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    valid_mask = block_indices < xnumel
    
    index_mod_96 = block_indices % 96
    index_div_96_mod_28 = (block_indices // 96) % 28
    index_div_2688 = block_indices // 2688
    index_div_96 = block_indices // 96
    
    input_offset = 5376 + index_mod_96 + 192 * index_div_96_mod_28 + 10752 * index_div_2688
    output_offset = index_mod_96 + 384 * index_div_96
    
    temp_data = tl.load(in_ptr0 + input_offset, valid_mask)
    tl.store(out_ptr0 + output_offset, temp_data, valid_mask)