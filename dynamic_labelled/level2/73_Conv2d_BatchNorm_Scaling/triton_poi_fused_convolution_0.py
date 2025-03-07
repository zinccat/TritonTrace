# From: 73_Conv2d_BatchNorm_Scaling

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_0poi_fused_convolution_0(in_out_ptr, input_ptr, kernel_size, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    indices = offset + tl.arange(0, XBLOCK)[:]
    mask = indices < num_elements
    linear_index = indices
    batch_index = ((indices // kernel_size) % 16)
    
    output_data = tl.load(in_out_ptr + (linear_index), mask, eviction_policy='evict_last')
    input_data = tl.load(input_ptr + (batch_index), mask, eviction_policy='evict_last')
    
    result_data = output_data + input_data
    tl.store(in_out_ptr + (linear_index), result_data, mask)