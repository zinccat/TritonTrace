# From: 72_ConvTranspose3d_BatchNorm_AvgPool_AvgPool

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_convolution_0(output_ptr, input_ptr, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    linear_index = block_indices
    channel_index = (block_indices // 250047) % 16
    
    output_value = tl.load(output_ptr + (linear_index), None)
    input_value = tl.load(input_ptr + (channel_index), None, eviction_policy='evict_last')
    
    result_value = output_value + input_value
    tl.store(output_ptr + (linear_index), result_value, None)