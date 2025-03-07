# From: 10_ConvTranspose2d_MaxPool_Hardtanh_Mean_Tanh

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_0poi_fused_convolution_0(in_out_ptr0, in_ptr0, kernel_size, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    indices = offset + tl.arange(0, XBLOCK)[:]
    mask = indices < num_elements
    output_indices = indices
    input_indices = ((indices // kernel_size) % 64)
    
    output_values = tl.load(in_out_ptr0 + (output_indices), mask, eviction_policy='evict_last')
    input_values = tl.load(in_ptr0 + (input_indices), mask, eviction_policy='evict_last')
    
    result_values = output_values + input_values
    tl.store(in_out_ptr0 + (output_indices), result_values, mask)