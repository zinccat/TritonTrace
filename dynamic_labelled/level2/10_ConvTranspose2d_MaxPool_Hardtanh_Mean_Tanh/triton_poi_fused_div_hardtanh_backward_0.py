# From: 10_ConvTranspose2d_MaxPool_Hardtanh_Mean_Tanh

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_div_hardtanh_backward_0(input_grad_ptr, input_data_ptr, input_mask_ptr, output_grad_ptr, kernel_size, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    indices = offset + tl.arange(0, XBLOCK)[:]
    mask = indices < num_elements
    element_indices = indices
    input_indices = indices // kernel_size
    
    grad_mask = tl.load(input_grad_ptr + (element_indices), mask, eviction_policy='evict_last').to(tl.int1)
    input_data = tl.load(input_data_ptr + (input_indices), mask, eviction_policy='evict_last')
    input_mask = tl.load(input_mask_ptr + (input_indices), mask, eviction_policy='evict_last')
    
    mask_squared = input_mask * input_mask
    one = 1.0
    mask_complement = one - mask_squared
    
    scaled_grad = input_data * mask_complement
    kernel_size_float = kernel_size.to(tl.float32)
    normalized_grad = scaled_grad / kernel_size_float
    
    zero = 0.0
    final_grad = tl.where(grad_mask, zero, normalized_grad)
    
    tl.store(output_grad_ptr + (element_indices), final_grad, mask)