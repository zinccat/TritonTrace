# From: 30_SwinTransformerV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_23poi_fused_cat_23(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 752640
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    valid_mask = block_indices < xnumel
    
    element_within_block = block_indices % 96
    block_within_row = (block_indices // 96) % 28
    row_index = block_indices // 2688
    linear_block_index = block_indices // 96
    
    temp_value = tl.load(in_ptr0 + (96 + element_within_block + 192 * block_within_row + 10752 * row_index), valid_mask)
    tl.store(out_ptr0 + (element_within_block + 384 * linear_block_index), temp_value, valid_mask)