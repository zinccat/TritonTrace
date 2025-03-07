# From: 29_SwinMLP

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_10poi_fused_cat_10(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 752640
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    valid_mask = block_indices < xnumel
    
    element_index_within_block = block_indices % 96
    row_index_within_block = (block_indices // 96) % 28
    channel_index = block_indices // 2688
    linear_index_within_channel = block_indices // 96
    
    input_offset = element_index_within_block + 192 * row_index_within_block + 10752 * channel_index
    output_offset = element_index_within_block + 384 * linear_index_within_channel
    
    tmp0 = tl.load(in_ptr0 + input_offset, valid_mask)
    tl.store(out_ptr0 + output_offset, tmp0, valid_mask)