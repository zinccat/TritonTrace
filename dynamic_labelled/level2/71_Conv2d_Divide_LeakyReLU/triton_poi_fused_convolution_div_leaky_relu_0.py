# From: 71_Conv2d_Divide_LeakyReLU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_div_leaky_relu_0poi_fused_convolution_div_leaky_relu_0(
    in_out_ptr, input_ptr, output_ptr, kernel_size_0, kernel_size_1, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    global_index = index
    batch_index = ((index // kernel_size_0) % 16)
    
    in_out_value = tl.load(in_out_ptr + (global_index), mask, eviction_policy='evict_last')
    input_value = tl.load(input_ptr + (batch_index), mask, eviction_policy='evict_last')
    
    sum_value = in_out_value + input_value
    divisor = kernel_size_1.to(tl.float32)
    
    divided_value = sum_value / divisor
    zero = 0.0
    positive_mask = divided_value > zero
    
    leaky_slope = 0.01
    leaky_value = divided_value * leaky_slope
    
    leaky_relu_value = tl.where(positive_mask, divided_value, leaky_value)
    
    tl.store(output_ptr + (global_index), positive_mask, mask)
    tl.store(in_out_ptr + (global_index), leaky_relu_value, mask)