# From: 30_SwinTransformerV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_45poi_fused_cat_45(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 376320
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    valid_mask = block_indices < xnumel
    
    element_index = block_indices % 192
    channel_index = (block_indices // 192) % 14
    batch_index = block_indices // 2688
    batch_channel_index = block_indices // 192
    
    input_offset = 5376 + element_index + 384 * channel_index + 10752 * batch_index
    output_offset = element_index + 768 * batch_channel_index
    
    temp_data = tl.load(in_ptr0 + input_offset, valid_mask)
    tl.store(out_ptr0 + output_offset, temp_data, valid_mask)