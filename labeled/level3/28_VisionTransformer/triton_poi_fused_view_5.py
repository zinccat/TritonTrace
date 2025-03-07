# From: 28_VisionTransformer

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_view_5poi_fused_view_5(input_ptr, output_ptr, total_elements, BLOCK_SIZE : tl.constexpr):
    total_elements = 201728
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < total_elements
    element_indices = block_indices
    temp_data = tl.load(input_ptr + (201728 + element_indices), valid_mask)
    tl.store(output_ptr + (element_indices), temp_data, valid_mask)