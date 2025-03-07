# From: 96_ConvTranspose3d_Multiply_Max_GlobalAvgPool_Clamp

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_mul_0poi_fused_convolution_mul_0(in_out_ptr0, in_ptr0, kernel_size, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    global_index = index
    channel_index = ((index // kernel_size) % 16)
    
    loaded_in_out = tl.load(in_out_ptr0 + (global_index), mask, eviction_policy='evict_last')
    loaded_in = tl.load(in_ptr0 + (channel_index), mask, eviction_policy='evict_last')
    
    added_values = loaded_in_out + loaded_in
    scale_factor = 0.5
    scaled_values = added_values * scale_factor
    
    tl.store(in_out_ptr0 + (global_index), scaled_values, mask)