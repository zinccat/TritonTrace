# From: 15_ConvTranspose3d_BatchNorm_Subtract

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_convolution_0(output_ptr, input_ptr, total_elements, BLOCK_SIZE : tl.constexpr):
    total_elements = 62995968
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < total_elements
    linear_index = block_indices
    channel_index = (block_indices // 123039) % 32
    output_value = tl.load(output_ptr + (linear_index), valid_mask)
    input_value = tl.load(input_ptr + (channel_index), valid_mask, eviction_policy='evict_last')
    result_value = output_value + input_value
    tl.store(output_ptr + (linear_index), result_value, valid_mask)