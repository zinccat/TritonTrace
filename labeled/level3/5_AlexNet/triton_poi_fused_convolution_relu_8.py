# From: 5_AlexNet

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_relu_8poi_fused_convolution_relu_8(output_ptr, input_ptr, num_elements, BLOCK_SIZE : tl.constexpr):
    num_elements = 1866240
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    linear_index = block_indices
    element_index = block_indices % 256
    output_value = tl.load(output_ptr + (linear_index), valid_mask)
    input_value = tl.load(input_ptr + (element_index), valid_mask, eviction_policy='evict_last')
    fused_value = output_value + input_value
    zero_value = tl.full([1], 0, tl.int32)
    relu_output = triton_helpers.maximum(zero_value, fused_value)
    tl.store(output_ptr + (linear_index), relu_output, valid_mask)