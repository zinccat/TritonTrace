# From: 27_Conv3d_HardSwish_ReLU_Softmax_Mean

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_convolution_0(output_ptr, input_ptr, num_elements, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    element_indices = block_indices
    channel_indices = (block_indices // 12600) % 16
    
    output_values = tl.load(output_ptr + (element_indices), None)
    input_values = tl.load(input_ptr + (channel_indices), None, eviction_policy='evict_last')
    
    result_values = output_values + input_values
    tl.store(output_ptr + (element_indices), result_values, None)