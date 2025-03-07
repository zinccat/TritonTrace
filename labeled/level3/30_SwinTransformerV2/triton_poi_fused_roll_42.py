# From: 30_SwinTransformerV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_roll_42poi_fused_roll_42(output_ptr, num_elements, BLOCK_SIZE : tl.constexpr):
    num_elements = 28
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    original_indices = block_indices
    rolled_indices = ((25 + original_indices) % 28)
    tl.store(output_ptr + (original_indices), rolled_indices, valid_mask)