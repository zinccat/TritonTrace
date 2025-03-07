# From: 28_VisionTransformer

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_view_1poi_fused_view_1(input_ptr, output_ptr, num_elements, BLOCK_SIZE : tl.constexpr):
    num_elements = 301056
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    element_index_within_block = block_indices % 768
    block_index_within_layer = block_indices // 768
    linear_index = block_indices
    tmp0 = tl.load(input_ptr + (element_index_within_block + 768 * block_index_within_layer + 150528 * ((element_index_within_block + 768 * (block_index_within_layer % 196)) // 150528)), valid_mask)
    tl.store(output_ptr + (linear_index), tmp0, valid_mask)