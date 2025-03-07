# From: 74_ConvTranspose3d_LeakyReLU_Multiply_LeakyReLU_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_leaky_relu_leaky_relu_backward_mul_2poi_fused_leaky_relu_leaky_relu_backward_mul_2(
    in_out_ptr0, in_ptr0, in_ptr1, kernel_size, xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    # Load data from pointers
    grad_output = tl.load(in_out_ptr0 + (xindex), None, eviction_policy='evict_last')
    weight = tl.load(in_ptr0 + (xindex // kernel_size) % 32, None, eviction_policy='evict_last')
    input_data = tl.load(in_ptr1 + (xindex), None, eviction_policy='evict_last')
    
    # Leaky ReLU backward pass
    zero = 0.0
    negative_slope = 0.2
    grad_output_leaky = tl.where(grad_output > zero, grad_output, grad_output * negative_slope)
    
    # Element-wise multiplication
    grad_weight = grad_output_leaky * weight
    
    # Second Leaky ReLU backward pass
    grad_weight_leaky = tl.where(grad_weight > zero, input_data, input_data * negative_slope)
    grad_input = grad_weight_leaky * weight * negative_slope
    
    # Store the result
    grad_input_final = tl.where(grad_output > zero, grad_weight, grad_input)
    tl.store(in_out_ptr0 + (xindex), grad_input_final, None)