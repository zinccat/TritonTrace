# From: 7_GoogleNetInceptionV1

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_33poi_fused_convolution_33(output_ptr, input_ptr, num_elements, BLOCK_SIZE : tl.constexpr):
    num_elements = 31360
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    linear_index = block_indices
    element_index = block_indices % 16
    output_data = tl.load(output_ptr + (linear_index), valid_mask)
    input_data = tl.load(input_ptr + (element_index), valid_mask, eviction_policy='evict_last')
    result_data = output_data + input_data
    tl.store(output_ptr + (linear_index), result_data, valid_mask)