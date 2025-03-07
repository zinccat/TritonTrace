# From: 91_ConvTranspose2d_Softmax_BiasAdd_Scaling_Sigmoid

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_0poi_fused_convolution_0(output_ptr, input_ptr, kernel_size, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    indices = offset + tl.arange(0, XBLOCK)[:]
    mask = indices < num_elements
    output_indices = indices
    input_indices = ((indices // kernel_size) % 64)
    
    output_values = tl.load(output_ptr + (output_indices), mask, eviction_policy='evict_last')
    input_values = tl.load(input_ptr + (input_indices), mask, eviction_policy='evict_last')
    
    result_values = output_values + input_values
    tl.store(output_ptr + (output_indices), result_values, mask)