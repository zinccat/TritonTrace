# From: 30_SwinTransformerV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_16poi_fused_clone_16(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    channel_index = block_indices % 96
    height_index_1 = (block_indices // 96) % 7
    height_index_2 = (block_indices // 672) % 7
    width_index_1 = (block_indices // 4704) % 8
    width_index_2 = (block_indices // 37632) % 8
    batch_index = block_indices // 301056
    linear_index = block_indices
    
    offset_1 = channel_index + 96 * (((3 + height_index_1 + 7 * width_index_1) % 56))
    offset_2 = 5376 * (((3 + height_index_2 + 7 * width_index_2) % 56))
    total_offset = offset_1 + offset_2 + 301056 * batch_index
    
    tmp0 = tl.load(in_ptr0 + total_offset, None)
    tl.store(out_ptr0 + (linear_index), tmp0, None)