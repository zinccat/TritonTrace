# From: 48_Conv3d_Scaling_Tanh_Multiply_Sigmoid

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_mul_sigmoid_backward_tanh_tanh_backward_0poi_fused_mul_sigmoid_backward_tanh_tanh_backward_0(
    input_grad_output_ptr, input_output_ptr, grad_input_weight_ptr, input_weight_ptr, grad_input_bias_ptr, 
    output_grad_input_ptr, output_grad_weight_ptr, kernel_size, num_elements, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < num_elements
    linear_index = xindex
    weight_index = ((xindex // kernel_size) % 16)
    
    grad_output = tl.load(input_grad_output_ptr + (linear_index), xmask, eviction_policy='evict_last')
    output = tl.load(input_output_ptr + (linear_index), xmask, eviction_policy='evict_last')
    grad_input_weight = tl.load(grad_input_weight_ptr + (weight_index), xmask, eviction_policy='evict_last')
    input_weight = tl.load(input_weight_ptr + (linear_index), xmask, eviction_policy='evict_last')
    grad_input_bias = tl.load(grad_input_bias_ptr + (weight_index), xmask, eviction_policy='evict_last')
    
    one = 1.0
    one_minus_output = one - output
    output_derivative = output * one_minus_output
    grad_input = grad_output * output_derivative
    grad_input_weighted = grad_input * grad_input_weight
    
    tanh_input = input_weight * grad_input_bias
    tanh_output = tl.extra.cuda.libdevice.tanh(tanh_input)
    tanh_derivative = tanh_output * tanh_output
    one_minus_tanh_derivative = one - tanh_derivative
    
    grad_input_weighted_tanh = grad_input_weighted * one_minus_tanh_derivative
    grad_input_weighted_tanh_bias = grad_input_weighted_tanh * grad_input_bias
    
    tl.store(output_grad_input_ptr + (linear_index), grad_input_weighted_tanh, xmask)
    tl.store(output_grad_weight_ptr + (linear_index), grad_input_weighted_tanh_bias, xmask)