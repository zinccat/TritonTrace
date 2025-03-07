# From: 30_SwinTransformerV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_24poi_fused_cat_24(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 752640
    program_id_offset = tl.program_id(0) * XBLOCK
    index_within_block = program_id_offset + tl.arange(0, XBLOCK)[:]
    valid_mask = index_within_block < xnumel
    
    block_offset = index_within_block % 96
    row_within_block = (index_within_block // 96) % 28
    block_index = index_within_block // 2688
    linear_index = index_within_block // 96
    
    input_address = 5472 + block_offset + 192 * row_within_block + 10752 * block_index
    output_address = block_offset + 384 * linear_index
    
    tmp0 = tl.load(in_ptr0 + input_address, valid_mask)
    tl.store(out_ptr0 + output_address, tmp0, valid_mask)