# From: 10_ConvTranspose2d_MaxPool_Hardtanh_Mean_Tanh

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_convolution_0(output_ptr, input_ptr, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    element_indices = block_indices
    channel_indices = (block_indices // 1024) % 64
    
    output_values = tl.load(output_ptr + (element_indices), None)
    input_values = tl.load(input_ptr + (channel_indices), None, eviction_policy='evict_last')
    
    result_values = output_values + input_values
    tl.store(output_ptr + (element_indices), result_values, None)