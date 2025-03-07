# From: 30_SwinTransformerV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_roll_65poi_fused_roll_65(out_ptr0, total_elements, BLOCK_SIZE : tl.constexpr):
    total_elements = 14
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < total_elements
    original_indices = block_indices
    rolled_indices = ((11 + original_indices) % total_elements)
    tl.store(out_ptr0 + (original_indices), rolled_indices, valid_mask)