# From: 83_Conv3d_GroupNorm_Min_Clamp_Dropout

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_1poi_fused_convolution_1(in_out_ptr0, in_ptr0, kernel_size, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    indices = offset + tl.arange(0, XBLOCK)[:]
    mask = indices < num_elements
    linear_index = indices
    group_index = ((indices // kernel_size) % 16)
    
    output_data = tl.load(in_out_ptr0 + (linear_index), mask, eviction_policy='evict_last')
    input_data = tl.load(in_ptr0 + (group_index), mask, eviction_policy='evict_last')
    
    result_data = output_data + input_data
    tl.store(in_out_ptr0 + (linear_index), result_data, mask)