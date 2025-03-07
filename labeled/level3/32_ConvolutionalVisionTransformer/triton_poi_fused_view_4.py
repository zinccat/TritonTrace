# From: 32_ConvolutionalVisionTransformer

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_view_4poi_fused_view_4(input_ptr, output_ptr, num_elements, BLOCK_SIZE : tl.constexpr):
    num_elements = 2560
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    indices = block_indices
    temp_data = tl.load(input_ptr + (2560 + indices), valid_mask)
    tl.store(output_ptr + (indices), temp_data, valid_mask)