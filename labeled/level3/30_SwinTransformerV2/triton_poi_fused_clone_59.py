# From: 30_SwinTransformerV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_59poi_fused_clone_59(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 752640
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    valid_mask = block_indices < xnumel
    
    element_within_block = block_indices % 32
    block_within_row = (block_indices // 32) % 12
    row_within_grid = (block_indices // 384) % 49
    grid_index = block_indices // 18816
    linear_index = block_indices
    
    tmp0 = tl.load(in_ptr0 + (element_within_block + 32 * row_within_grid + 1568 * block_within_row + 18816 * grid_index), valid_mask)
    tl.store(out_ptr0 + (linear_index), tmp0, valid_mask)