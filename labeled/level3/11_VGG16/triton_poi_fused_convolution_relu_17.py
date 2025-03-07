# From: 11_VGG16

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_relu_17poi_fused_convolution_relu_17(output_ptr, input_ptr, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    global_indices = block_indices
    local_indices = (block_indices % 512)
    
    output_value = tl.load(output_ptr + (global_indices), None)
    input_value = tl.load(input_ptr + (local_indices), None, eviction_policy='evict_last')
    
    fused_value = output_value + input_value
    zero_value = tl.full([1], 0, tl.int32)
    relu_output = triton_helpers.maximum(zero_value, fused_value)
    
    tl.store(output_ptr + (global_indices), relu_output, None)