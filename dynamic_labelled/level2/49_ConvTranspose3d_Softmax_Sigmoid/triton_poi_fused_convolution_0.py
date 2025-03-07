# From: 49_ConvTranspose3d_Softmax_Sigmoid

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_0poi_fused_convolution_0(output_ptr, input_ptr, kernel_size, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    indices = offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    output_index = indices
    input_index = ((indices // kernel_size) % 64)
    
    output_value = tl.load(output_ptr + (output_index), None, eviction_policy='evict_last')
    input_value = tl.load(input_ptr + (input_index), None, eviction_policy='evict_last')
    
    result_value = output_value + input_value
    tl.store(output_ptr + (output_index), result_value, None)