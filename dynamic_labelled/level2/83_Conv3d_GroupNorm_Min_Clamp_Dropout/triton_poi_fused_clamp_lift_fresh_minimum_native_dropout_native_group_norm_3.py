# From: 83_Conv3d_GroupNorm_Min_Clamp_Dropout

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clamp_lift_fresh_minimum_native_dropout_native_group_norm_3(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, input_ptr5, 
    output_ptr0, kernel_size0, kernel_size1, num_elements, XBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    x_mask = x_index < num_elements
    x3 = x_index
    x5 = x_index // kernel_size0
    x1 = ((x_index // kernel_size1) % 16)
    
    input_masked = tl.load(input_ptr0 + (x3), x_mask, eviction_policy='evict_last').to(tl.int1)
    input_data1 = tl.load(input_ptr1 + (x3), x_mask, eviction_policy='evict_last')
    input_data2 = tl.load(input_ptr2 + (x5 // 2), x_mask, eviction_policy='evict_last')
    input_data3 = tl.load(input_ptr3 + (x5 // 2), x_mask, eviction_policy='evict_last')
    input_data4 = tl.load(input_ptr4 + (x1), x_mask, eviction_policy='evict_last')
    input_data5 = tl.load(input_ptr5 + (x1), x_mask, eviction_policy='evict_last')
    
    float_input = input_masked.to(tl.float32)
    subtracted = input_data1 - input_data2
    multiplied1 = subtracted * input_data3
    multiplied2 = multiplied1 * input_data4
    added = multiplied2 + input_data5
    
    min_value = 0.0
    clamped_min = triton_helpers.minimum(added, min_value)
    clamped_max = triton_helpers.maximum(clamped_min, min_value)
    clamped = triton_helpers.minimum(clamped_max, 1.0)
    
    scaled_output = float_input * clamped
    final_output = scaled_output * 1.25
    
    tl.store(output_ptr0 + (x3), final_output, x_mask)