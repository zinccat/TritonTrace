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
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    linear_index = index
    group_index = index // kernel_size0
    channel_index = ((index // kernel_size1) % 16)
    
    input_masked = tl.load(input_ptr0 + (linear_index), mask, eviction_policy='evict_last').to(tl.int1)
    input_data = tl.load(input_ptr1 + (linear_index), mask, eviction_policy='evict_last')
    group_mean = tl.load(input_ptr2 + (group_index // 2), mask, eviction_policy='evict_last')
    group_var = tl.load(input_ptr3 + (group_index // 2), mask, eviction_policy='evict_last')
    gamma = tl.load(input_ptr4 + (channel_index), mask, eviction_policy='evict_last')
    beta = tl.load(input_ptr5 + (channel_index), mask, eviction_policy='evict_last')
    
    normalized_data = input_data - group_mean
    scaled_data = normalized_data * group_var
    gamma_scaled = scaled_data * gamma
    batch_norm_output = gamma_scaled + beta
    
    clamped_output = triton_helpers.minimum(batch_norm_output, 0.0)
    clamped_output = triton_helpers.maximum(clamped_output, 0.0)
    clamped_output = triton_helpers.minimum(clamped_output, 1.0)
    
    dropout_output = input_masked.to(tl.float32) * clamped_output
    scaled_dropout_output = dropout_output * 1.25
    
    tl.store(output_ptr0 + (linear_index), scaled_dropout_output, mask)