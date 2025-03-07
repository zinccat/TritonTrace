# From: 2_ConvTranspose2d_BiasAdd_Clamp_Scaling_Clamp_Divide

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_clamp_convolution_div_ge_le_logical_and_mul_0poi_fused_add_clamp_convolution_div_ge_le_logical_and_mul_0(
    input_ptr0, input_ptr1, input_ptr2, output_ptr0, output_ptr1, output_ptr2, kernel_size, num_elements, XBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    x_mask = x_index < num_elements
    x3 = x_index
    x1 = ((x_index // kernel_size) % 16)
    
    input_data0 = tl.load(input_ptr0 + (x3), x_mask, eviction_policy='evict_last')
    input_data1 = tl.load(input_ptr1 + (x1), x_mask, eviction_policy='evict_last')
    input_data2 = tl.load(input_ptr2 + (x1), x_mask, eviction_policy='evict_last')
    
    sum_data = input_data0 + input_data1
    total_sum = sum_data + input_data2
    
    clamp_min = 0.0
    max_clamped = triton_helpers.maximum(total_sum, clamp_min)
    clamp_max = 1.0
    min_clamped = triton_helpers.minimum(max_clamped, clamp_max)
    
    scale_factor = 2.0
    scaled_value = min_clamped * scale_factor
    scaled_clamped_min = triton_helpers.maximum(scaled_value, clamp_min)
    scaled_clamped_max = triton_helpers.minimum(scaled_clamped_min, clamp_max)
    
    final_value = scaled_clamped_max * 0.5
    
    within_scaled_bounds = (scaled_value >= clamp_min) & (scaled_value <= clamp_max)
    within_original_bounds = (total_sum >= clamp_min) & (total_sum <= clamp_max)
    
    tl.store(output_ptr0 + (x3), final_value, x_mask)
    tl.store(output_ptr1 + (x3), within_scaled_bounds, x_mask)
    tl.store(output_ptr2 + (x3), within_original_bounds, x_mask)