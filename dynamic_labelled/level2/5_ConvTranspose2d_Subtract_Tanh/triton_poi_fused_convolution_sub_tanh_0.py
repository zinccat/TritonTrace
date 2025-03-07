# From: 5_ConvTranspose2d_Subtract_Tanh

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_sub_tanh_0poi_fused_convolution_sub_tanh_0(
    output_ptr, input_ptr1, input_ptr2, kernel_size, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    indices = offset + tl.arange(0, XBLOCK)[:]
    mask = indices < num_elements
    linear_index = indices
    channel_index = ((indices // kernel_size) % 16)
    
    output_values = tl.load(output_ptr + (linear_index), mask, eviction_policy='evict_last')
    input_values1 = tl.load(input_ptr1 + (channel_index), mask, eviction_policy='evict_last')
    input_values2 = tl.load(input_ptr2 + (channel_index), mask, eviction_policy='evict_last')
    
    added_values = output_values + input_values1
    subtracted_values = added_values - input_values2
    tanh_values = tl.extra.cuda.libdevice.tanh(subtracted_values)
    
    tl.store(output_ptr + (linear_index), tanh_values, mask)