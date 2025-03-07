# From: 54_Conv2d_Multiply_LeakyReLU_GELU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_gelu_leaky_relu_mul_0poi_fused_convolution_gelu_leaky_relu_mul_0(
    in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, kernel_size, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    linear_index = index
    channel_index = ((index // kernel_size) % 16)
    
    input_output_value = tl.load(in_out_ptr0 + (linear_index), mask, eviction_policy='evict_last')
    input_value_0 = tl.load(in_ptr0 + (channel_index), mask, eviction_policy='evict_last')
    input_value_1 = tl.load(in_ptr1 + (channel_index), mask, eviction_policy='evict_last')
    
    add_result = input_output_value + input_value_0
    multiply_result = add_result * input_value_1
    
    zero = 0.0
    greater_than_zero = multiply_result > zero
    leaky_relu_slope = 0.01
    leaky_relu_result = multiply_result * leaky_relu_slope
    
    gelu_intermediate = tl.where(greater_than_zero, multiply_result, leaky_relu_result)
    gelu_half = 0.5
    gelu_scaled = gelu_intermediate * gelu_half
    
    gelu_coefficient = 0.7071067811865476
    gelu_argument = gelu_intermediate * gelu_coefficient
    erf_result = tl.extra.cuda.libdevice.erf(gelu_argument)
    
    erf_offset = 1.0
    erf_sum = erf_result + erf_offset
    gelu_final = gelu_scaled * erf_sum
    
    tl.store(in_out_ptr0 + (linear_index), add_result, mask)
    tl.store(out_ptr0 + (linear_index), gelu_final, mask)