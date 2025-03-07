# From: 30_SwinTransformerV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_2poi_fused_clone_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    col_index = block_indices % 672
    row_index = (block_indices // 672) % 7
    depth_index = (block_indices // 4704) % 8
    batch_index = block_indices // 37632
    linear_index = block_indices
    
    input_offset = col_index + 672 * depth_index + 5376 * row_index + 37632 * batch_index
    tmp0 = tl.load(in_ptr0 + input_offset, None)
    tl.store(out_ptr0 + linear_index, tmp0, None)