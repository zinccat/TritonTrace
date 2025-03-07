# From: 43_Conv3d_Max_LogSumExp_ReLU

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_convolution_0(output_ptr, input_ptr, num_elements, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    element_index = block_indices
    channel_index = (block_indices // 16384) % 16
    
    output_value = tl.load(output_ptr + (element_index), None)
    input_value = tl.load(input_ptr + (channel_index), None, eviction_policy='evict_last')
    
    result_value = output_value + input_value
    tl.store(output_ptr + (element_index), result_value, None)