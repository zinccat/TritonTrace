# From: 74_ConvTranspose3d_LeakyReLU_Multiply_LeakyReLU_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_convolution_leaky_relu_mul_0(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, kernel_size, num_elements, XBLOCK : tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x3 = x_index
    x1 = ((x_index // kernel_size) % 32)
    
    # Load data from pointers
    input_output_value = tl.load(in_out_ptr0 + (x3), None, eviction_policy='evict_last')
    input_value_0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    input_value_1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    
    # Perform addition
    added_value = input_output_value + input_value_0
    
    # Leaky ReLU parameters
    zero = 0.0
    leaky_relu_slope = 0.2
    
    # Apply Leaky ReLU
    is_positive = added_value > zero
    leaky_relu_value = added_value * leaky_relu_slope
    activated_value = tl.where(is_positive, added_value, leaky_relu_value)
    
    # Multiply with second input
    multiplied_value = activated_value * input_value_1
    
    # Apply Leaky ReLU again
    is_positive_after_mul = multiplied_value > zero
    leaky_relu_value_after_mul = multiplied_value * leaky_relu_slope
    final_activated_value = tl.where(is_positive_after_mul, multiplied_value, leaky_relu_value_after_mul)
    
    # Store results
    tl.store(in_out_ptr0 + (x3), added_value, None)
    tl.store(out_ptr0 + (x3), final_activated_value, None)