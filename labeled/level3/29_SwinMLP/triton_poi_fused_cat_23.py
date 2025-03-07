# From: 29_SwinMLP

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_23poi_fused_cat_23(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 376320
    block_id = tl.program_id(0) * XBLOCK
    index_within_block = tl.arange(0, XBLOCK)[:]
    valid_indices_mask = index_within_block < xnumel
    
    element_within_row = index_within_block % 192
    row_index = (index_within_block // 192) % 14
    block_index = index_within_block // 2688
    row_in_block = index_within_block // 192
    
    input_offset = element_within_row + 384 * row_index + 10752 * block_index
    output_offset = element_within_row + 768 * row_in_block
    
    temp_data = tl.load(in_ptr0 + input_offset, valid_indices_mask)
    tl.store(out_ptr0 + output_offset, temp_data, valid_indices_mask)