# From: 87_Conv2d_Subtract_Subtract_Mish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_0(output_ptr, input_ptr, kernel_size, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    indices = offset + tl.arange(0, XBLOCK)[:]
    mask = indices < num_elements
    linear_index = indices
    channel_index = ((indices // kernel_size) % 16)
    
    output_value = tl.load(output_ptr + (linear_index), mask, eviction_policy='evict_last')
    input_value = tl.load(input_ptr + (channel_index), mask, eviction_policy='evict_last')
    result_value = output_value + input_value
    
    tl.store(output_ptr + (linear_index), result_value, mask)