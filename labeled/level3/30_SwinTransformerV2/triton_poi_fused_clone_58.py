# From: 30_SwinTransformerV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_58poi_fused_clone_58(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 752640
    program_id_offset = tl.program_id(0) * XBLOCK
    index_within_block = program_id_offset + tl.arange(0, XBLOCK)[:]
    valid_mask = index_within_block < xnumel
    
    block_offset = index_within_block % 32
    block_row = (index_within_block // 32) % 49
    block_depth = (index_within_block // 1568) % 12
    block_slice = index_within_block // 18816
    linear_index = index_within_block
    
    offset = 768 + block_offset + 32 * block_depth + 1152 * block_row + 56448 * block_slice
    tmp0 = tl.load(in_ptr0 + offset, valid_mask)
    tl.store(out_ptr0 + linear_index, tmp0, valid_mask)