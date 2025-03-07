# From: 10_ConvTranspose2d_MaxPool_Hardtanh_Mean_Tanh

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_0(output_ptr, input_ptr, kernel_size, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    output_indices = block_indices
    input_indices = ((block_indices // kernel_size) % 64)
    
    output_values = tl.load(output_ptr + (output_indices), valid_mask, eviction_policy='evict_last')
    input_values = tl.load(input_ptr + (input_indices), valid_mask, eviction_policy='evict_last')
    
    result_values = output_values + input_values
    tl.store(output_ptr + (output_indices), result_values, valid_mask)