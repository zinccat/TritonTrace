# From: 71_Conv2d_Divide_LeakyReLU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_div_leaky_relu_0(in_out_ptr, input_ptr, output_ptr, kernel_size_x, kernel_size_y, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    indices = offset + tl.arange(0, XBLOCK)[:]
    mask = indices < num_elements
    linear_index = indices
    channel_index = ((indices // kernel_size_x) % 16)
    
    in_out_data = tl.load(in_out_ptr + (linear_index), mask, eviction_policy='evict_last')
    input_data = tl.load(input_ptr + (channel_index), mask, eviction_policy='evict_last')
    
    sum_data = in_out_data + input_data
    divisor = kernel_size_y.to(tl.float32)
    
    divided_data = sum_data / divisor
    zero = 0.0
    positive_mask = divided_data > zero
    
    leaky_slope = 0.01
    leaky_data = divided_data * leaky_slope
    
    leaky_relu_result = tl.where(positive_mask, divided_data, leaky_data)
    
    tl.store(output_ptr + (linear_index), positive_mask, mask)
    tl.store(in_out_ptr + (linear_index), leaky_relu_result, mask)