# From: 2_ConvTranspose2d_BiasAdd_Clamp_Scaling_Clamp_Divide

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_clamp_convolution_div_ge_le_logical_and_mul_0(
    input_ptr0, input_ptr1, input_ptr2, output_ptr0, output_ptr1, output_ptr2, kernel_size, num_elements, XBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    x_mask = x_index < num_elements
    x3 = x_index
    x1 = ((x_index // kernel_size) % 16)
    
    input_val0 = tl.load(input_ptr0 + (x3), x_mask, eviction_policy='evict_last')
    input_val1 = tl.load(input_ptr1 + (x1), x_mask, eviction_policy='evict_last')
    input_val2 = tl.load(input_ptr2 + (x1), x_mask, eviction_policy='evict_last')
    
    sum_val = input_val0 + input_val1
    total_sum = sum_val + input_val2
    
    clamp_min = 0.0
    clamp_max = 1.0
    
    clamped_val = triton_helpers.maximum(total_sum, clamp_min)
    clamped_val = triton_helpers.minimum(clamped_val, clamp_max)
    
    scaled_val = clamped_val * 2.0
    scaled_clamped_val = triton_helpers.maximum(scaled_val, clamp_min)
    final_clamped_val = triton_helpers.minimum(scaled_clamped_val, clamp_max)
    
    scaled_down_val = final_clamped_val * 0.5
    
    is_within_bounds = (scaled_val >= clamp_min) & (scaled_val <= clamp_max)
    is_original_within_bounds = (total_sum >= clamp_min) & (total_sum <= clamp_max)
    
    tl.store(output_ptr0 + (x3), scaled_down_val, x_mask)
    tl.store(output_ptr1 + (x3), is_within_bounds, x_mask)
    tl.store(output_ptr2 + (x3), is_original_within_bounds, x_mask)