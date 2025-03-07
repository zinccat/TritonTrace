# From: 30_SwinTransformerV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_81poi_fused_clone_81(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 376320
    program_id_offset = tl.program_id(0) * XBLOCK
    index_within_block = program_id_offset + tl.arange(0, XBLOCK)[:]
    valid_mask = index_within_block < xnumel
    
    block_offset = index_within_block % 32
    row_index = (index_within_block // 32) % 24
    depth_index = (index_within_block // 768) % 49
    batch_index = index_within_block // 37632
    linear_index = index_within_block
    
    input_address = in_ptr0 + (block_offset + 32 * depth_index + 1568 * row_index + 37632 * batch_index)
    tmp0 = tl.load(input_address, valid_mask)
    tl.store(out_ptr0 + (linear_index), tmp0, valid_mask)