# From: 28_VisionTransformer

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_0poi_fused_clone_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 301056
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    valid_mask = block_indices < xnumel
    
    col_index = block_indices % 16
    row_index = (block_indices // 16) % 16
    depth_index = (block_indices // 256) % 14
    batch_index = block_indices // 3584
    linear_index = block_indices
    
    tmp0 = tl.load(in_ptr0 + (col_index + 16 * depth_index + 224 * row_index + 3584 * batch_index), valid_mask)
    tl.store(out_ptr0 + (linear_index), tmp0, valid_mask)