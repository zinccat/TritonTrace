# From: 29_SwinMLP

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_42poi_fused_cat_42(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 188160
    program_id_offset = tl.program_id(0) * XBLOCK
    index_within_block = program_id_offset + tl.arange(0, XBLOCK)[:]
    valid_mask = index_within_block < xnumel
    
    block_offset = index_within_block % 384
    block_row = (index_within_block // 384) % 7
    block_depth = index_within_block // 2688
    block_index = index_within_block // 384
    
    input_offset = 5760 + block_offset + 768 * block_row + 10752 * block_depth
    output_offset = block_offset + 1536 * block_index
    
    temp_value = tl.load(in_ptr0 + input_offset, valid_mask)
    tl.store(out_ptr0 + output_offset, temp_value, valid_mask)