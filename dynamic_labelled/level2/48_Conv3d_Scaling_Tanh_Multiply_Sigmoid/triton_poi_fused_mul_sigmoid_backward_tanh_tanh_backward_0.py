# From: 48_Conv3d_Scaling_Tanh_Multiply_Sigmoid

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_mul_sigmoid_backward_tanh_tanh_backward_0(
    input_grad_output_ptr, input_output_ptr, input_weight_ptr, input_input_ptr, input_weight_grad_ptr,
    grad_input_ptr, grad_weight_ptr, kernel_size, num_elements, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < num_elements
    x3 = xindex
    x1 = ((xindex // kernel_size) % 16)
    
    grad_output = tl.load(input_grad_output_ptr + (x3), xmask, eviction_policy='evict_last')
    output = tl.load(input_output_ptr + (x3), xmask, eviction_policy='evict_last')
    weight = tl.load(input_weight_ptr + (x1), xmask, eviction_policy='evict_last')
    input = tl.load(input_input_ptr + (x3), xmask, eviction_policy='evict_last')
    weight_grad = tl.load(input_weight_grad_ptr + (x1), xmask, eviction_policy='evict_last')
    
    one = 1.0
    one_minus_output = one - output
    output_derivative = output * one_minus_output
    grad_input = grad_output * output_derivative
    grad_weight = grad_input * weight
    
    input_weight_product = input * weight_grad
    tanh_result = tl.extra.cuda.libdevice.tanh(input_weight_product)
    tanh_squared = tanh_result * tanh_result
    one_minus_tanh_squared = one - tanh_squared
    
    grad_weight_tanh = grad_weight * one_minus_tanh_squared
    grad_weight_final = grad_weight_tanh * weight_grad
    
    tl.store(grad_input_ptr + (x3), grad_weight, xmask)
    tl.store(grad_weight_ptr + (x3), grad_weight_final, xmask)