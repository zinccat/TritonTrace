# From: 32_ConvolutionalVisionTransformer

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_0poi_fused_convolution_0(in_out_ptr, input_ptr, num_elements, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    element_indices = block_indices
    channel_indices = (block_indices // 64) % 128
    
    output_value = tl.load(in_out_ptr + element_indices, None)
    input_value = tl.load(input_ptr + channel_indices, None, eviction_policy='evict_last')
    
    result_value = output_value + input_value
    tl.store(in_out_ptr + element_indices, result_value, None)