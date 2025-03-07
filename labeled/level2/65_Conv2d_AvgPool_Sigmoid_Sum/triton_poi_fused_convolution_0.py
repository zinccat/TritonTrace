# From: 65_Conv2d_AvgPool_Sigmoid_Sum

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_convolution_0(output_ptr, input_ptr, num_elements, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    element_indices = block_indices
    channel_indices = (block_indices // 900) % 16
    
    output_value = tl.load(output_ptr + (element_indices), None)
    input_value = tl.load(input_ptr + (channel_indices), None, eviction_policy='evict_last')
    
    result_value = output_value + input_value
    tl.store(output_ptr + (element_indices), result_value, None)