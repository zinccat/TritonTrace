# From: 74_ConvTranspose3d_LeakyReLU_Multiply_LeakyReLU_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_leaky_relu_leaky_relu_backward_mul_2(in_out_ptr0, in_ptr0, in_ptr1, kernel_size, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // kernel_size) % 32)
    
    grad_output = tl.load(in_out_ptr0 + (x3), None, eviction_policy='evict_last')
    input0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    input1 = tl.load(in_ptr1 + (x3), None, eviction_policy='evict_last')
    
    zero = 0.0
    leaky_relu_slope = 0.2
    
    grad_output_leaky_relu = tl.where(grad_output > zero, grad_output, grad_output * leaky_relu_slope)
    grad_input0 = grad_output_leaky_relu * input0
    
    grad_input0_leaky_relu = tl.where(grad_input0 > zero, input1, input1 * leaky_relu_slope)
    grad_input1 = grad_input0_leaky_relu * input0
    
    grad_input1_leaky_relu = grad_input1 * leaky_relu_slope
    final_grad_output = tl.where(grad_output > zero, grad_input1, grad_input1_leaky_relu)
    
    tl.store(in_out_ptr0 + (x3), final_grad_output, None)