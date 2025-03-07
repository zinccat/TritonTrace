# From: 52_Conv2d_Activation_BatchNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_0poi_fused_convolution_0(output_ptr, input_ptr, kernel_size, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    linear_index = block_indices
    channel_index = ((block_indices // kernel_size) % 16)
    
    output_values = tl.load(output_ptr + (linear_index), valid_mask, eviction_policy='evict_last')
    input_values = tl.load(input_ptr + (channel_index), valid_mask, eviction_policy='evict_last')
    
    result_values = output_values + input_values
    tl.store(output_ptr + (linear_index), result_values, valid_mask)