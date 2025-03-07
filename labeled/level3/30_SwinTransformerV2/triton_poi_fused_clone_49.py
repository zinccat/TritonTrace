# From: 30_SwinTransformerV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_49poi_fused_clone_49(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 752640
    program_id_offset = tl.program_id(0) * XBLOCK
    index_within_block = program_id_offset + tl.arange(0, XBLOCK)[:]
    valid_mask = index_within_block < xnumel
    
    block_position = index_within_block % 2688
    block_row = (index_within_block // 2688) % 7
    block_channel = (index_within_block // 18816) % 2
    block_depth = index_within_block // 37632
    linear_index = index_within_block
    
    input_offset = block_position + 2688 * block_channel + 5376 * block_row + 37632 * block_depth
    tmp0 = tl.load(in_ptr0 + input_offset, valid_mask)
    tl.store(out_ptr0 + linear_index, tmp0, valid_mask)