# From: 4_LeNet5

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_relu_0poi_fused_convolution_relu_0(output_ptr, input_ptr, num_elements, BLOCK_SIZE : tl.constexpr):
    num_elements = 4704
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    element_indices = block_indices
    batch_indices = block_indices // 784
    output_values = tl.load(output_ptr + (element_indices), valid_mask)
    input_values = tl.load(input_ptr + (batch_indices), valid_mask, eviction_policy='evict_last')
    fused_values = output_values + input_values
    zero_tensor = tl.full([1], 0, tl.int32)
    relu_values = triton_helpers.maximum(zero_tensor, fused_values)
    tl.store(output_ptr + (element_indices), relu_values, valid_mask)