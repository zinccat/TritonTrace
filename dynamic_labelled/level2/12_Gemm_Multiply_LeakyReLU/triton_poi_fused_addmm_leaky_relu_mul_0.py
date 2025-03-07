# From: 12_Gemm_Multiply_LeakyReLU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_addmm_leaky_relu_mul_0poi_fused_addmm_leaky_relu_mul_0(in_out_ptr, input_ptr, output_ptr, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    indices = offset + tl.arange(0, XBLOCK)[:]
    mask = indices < num_elements
    global_indices = indices
    local_indices = indices % 512
    
    in_out_values = tl.load(in_out_ptr + (global_indices), mask)
    input_values = tl.load(input_ptr + (local_indices), mask, eviction_policy='evict_last')
    
    added_values = in_out_values + input_values
    scaled_values = added_values * 2.0
    
    zero_threshold = 0.0
    is_positive = scaled_values > zero_threshold
    leaky_factor = 0.1
    leaky_values = scaled_values * leaky_factor
    
    activated_values = tl.where(is_positive, scaled_values, leaky_values)
    
    tl.store(output_ptr + (global_indices), is_positive, mask)
    tl.store(in_out_ptr + (global_indices), activated_values, mask)