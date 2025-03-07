# From: 30_SwinTransformerV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_69poi_fused_cat_69(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 188160
    program_id_offset = tl.program_id(0) * XBLOCK
    index_within_block = tl.arange(0, XBLOCK)[:]
    valid_indices_mask = index_within_block < xnumel
    
    index_within_channel = index_within_block % 384
    channel_index = (index_within_block // 384) % 7
    batch_index = index_within_block // 2688
    channel_block_index = index_within_block // 384
    
    input_offset = 384 + index_within_channel + 768 * channel_index + 10752 * batch_index
    output_offset = index_within_channel + 1536 * channel_block_index
    
    tmp0 = tl.load(in_ptr0 + input_offset, valid_indices_mask)
    tl.store(out_ptr0 + output_offset, tmp0, valid_indices_mask)