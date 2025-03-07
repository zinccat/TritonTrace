# From: 15_ConvTranspose3d_BatchNorm_Subtract

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_0poi_fused_convolution_0(in_out_ptr0, in_ptr0, kernel_size, num_elements, XBLOCK : tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    linear_index = index
    batch_index = ((index // kernel_size) % 32)
    
    output_value = tl.load(in_out_ptr0 + (linear_index), mask, eviction_policy='evict_last')
    input_value = tl.load(in_ptr0 + (batch_index), mask, eviction_policy='evict_last')
    result_value = output_value + input_value
    
    tl.store(in_out_ptr0 + (linear_index), result_value, mask)