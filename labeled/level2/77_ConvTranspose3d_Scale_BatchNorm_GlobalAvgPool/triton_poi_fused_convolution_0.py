# From: 77_ConvTranspose3d_Scale_BatchNorm_GlobalAvgPool

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_convolution_0(output_ptr, input_ptr, num_elements, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    global_index = block_indices
    batch_index = (block_indices // 20808) % 32
    
    output_value = tl.load(output_ptr + (global_index), None)
    input_value = tl.load(input_ptr + (batch_index), None, eviction_policy='evict_last')
    
    result_value = output_value + input_value
    tl.store(output_ptr + (global_index), result_value, None)