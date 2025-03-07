# From: 1_Conv2D_ReLU_BiasAdd

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_convolution_relu_threshold_backward_0(
    input_grad_ptr, weight_grad_ptr, bias_grad_ptr, output_grad_ptr, relu_mask_ptr, kernel_size, num_elements, XBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    x_mask = x_index < num_elements
    x3 = x_index
    x1 = ((x_index // kernel_size) % 16)
    
    input_grad = tl.load(input_grad_ptr + (x3), x_mask, eviction_policy='evict_last')
    weight_grad = tl.load(weight_grad_ptr + (x1), x_mask, eviction_policy='evict_last')
    bias_grad = tl.load(bias_grad_ptr + (x1), x_mask, eviction_policy='evict_last')
    
    grad_sum = input_grad + weight_grad
    zero_tensor = tl.full([1], 0, tl.int32)
    relu_output = triton_helpers.maximum(zero_tensor, grad_sum)
    bias_added_output = relu_output + bias_grad
    
    relu_threshold = 0.0
    relu_condition = relu_output <= relu_threshold
    
    tl.store(output_grad_ptr + (x3), bias_added_output, x_mask)
    tl.store(relu_mask_ptr + (x3), relu_condition, x_mask)