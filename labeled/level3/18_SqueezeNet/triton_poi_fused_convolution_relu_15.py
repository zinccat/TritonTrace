# From: 18_SqueezeNet

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_relu_15poi_fused_convolution_relu_15(output_ptr, input_ptr, num_elements, BLOCK_SIZE : tl.constexpr):
    num_elements = 23328
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    linear_index = block_indices
    element_index = block_indices % 32
    output_values = tl.load(output_ptr + (linear_index), valid_mask)
    input_values = tl.load(input_ptr + (element_index), valid_mask, eviction_policy='evict_last')
    accumulated_values = output_values + input_values
    zero_tensor = tl.full([1], 0, tl.int32)
    relu_output = triton_helpers.maximum(zero_tensor, accumulated_values)
    tl.store(output_ptr + (linear_index), relu_output, valid_mask)