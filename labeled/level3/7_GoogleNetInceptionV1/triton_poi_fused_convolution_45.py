# From: 7_GoogleNetInceptionV1

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_45poi_fused_convolution_45(output_ptr, input_ptr, num_elements, BLOCK_SIZE : tl.constexpr):
    num_elements = 313600
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    linear_index = block_indices
    channel_index = block_indices % 160
    output_value = tl.load(output_ptr + (linear_index), valid_mask)
    input_value = tl.load(input_ptr + (channel_index), valid_mask, eviction_policy='evict_last')
    result_value = output_value + input_value
    tl.store(output_ptr + (linear_index), result_value, valid_mask)