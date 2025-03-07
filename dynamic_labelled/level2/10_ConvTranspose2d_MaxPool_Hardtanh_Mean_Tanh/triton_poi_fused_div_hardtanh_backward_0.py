# From: 10_ConvTranspose2d_MaxPool_Hardtanh_Mean_Tanh

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_div_hardtanh_backward_0poi_fused_div_hardtanh_backward_0(input_ptr0, input_ptr1, input_ptr2, output_ptr0, kernel_size, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    element_index = index
    input_index = index // kernel_size
    
    input_value0 = tl.load(input_ptr0 + (element_index), mask, eviction_policy='evict_last').to(tl.int1)
    input_value1 = tl.load(input_ptr1 + (input_index), mask, eviction_policy='evict_last')
    input_value2 = tl.load(input_ptr2 + (input_index), mask, eviction_policy='evict_last')
    
    squared_value2 = input_value2 * input_value2
    one = 1.0
    subtracted_value = one - squared_value2
    multiplied_value = input_value1 * subtracted_value
    
    kernel_size_float = kernel_size.to(tl.float32)
    divided_value = multiplied_value / kernel_size_float
    
    zero = 0.0
    conditional_value = tl.where(input_value0, zero, divided_value)
    
    tl.store(output_ptr0 + (element_index), conditional_value, mask)