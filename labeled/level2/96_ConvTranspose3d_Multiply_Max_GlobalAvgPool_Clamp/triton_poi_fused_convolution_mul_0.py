# From: 96_ConvTranspose3d_Multiply_Max_GlobalAvgPool_Clamp

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_convolution_mul_0(input_ptr0, input_ptr1, output_ptr0, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    index = block_indices
    channel_index = (block_indices // 123039) % 16
    spatial_index = block_indices % 3969
    batch_index = (block_indices // 3969)
    
    input_value0 = tl.load(input_ptr0 + (index), None)
    input_value1 = tl.load(input_ptr1 + (channel_index), None, eviction_policy='evict_last')
    
    combined_value = input_value0 + input_value1
    scale_factor = 0.5
    scaled_value = combined_value * scale_factor
    
    output_index = spatial_index + (4000 * batch_index)
    tl.store(output_ptr0 + (output_index), scaled_value, None)