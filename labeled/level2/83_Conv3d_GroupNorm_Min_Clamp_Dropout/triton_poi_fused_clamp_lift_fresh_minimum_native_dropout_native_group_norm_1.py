# From: 83_Conv3d_GroupNorm_Min_Clamp_Dropout

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused_clamp_lift_fresh_minimum_native_dropout_native_group_norm_1(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, input_ptr5, 
    output_ptr1, output_ptr2, load_seed_offset, xnumel, XBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    x4 = x_index
    x0 = x_index % 12600
    x1 = (x_index // 12600)
    x2 = (x1 // 16)
    
    input_value1 = tl.load(input_ptr1 + (x4), None)
    input_value2 = tl.load(input_ptr2 + ((x1 // 2)), None, eviction_policy='evict_last')
    input_value3 = tl.load(input_ptr3 + ((x1 // 2)), None, eviction_policy='evict_last')
    input_value4 = tl.load(input_ptr4 + (x2), None, eviction_policy='evict_last')
    input_value5 = tl.load(input_ptr5 + (x2), None, eviction_policy='evict_last')
    
    seed_value = tl.load(input_ptr0 + load_seed_offset)
    random_index = x4
    random_value = tl.rand(seed_value, random_index.to(tl.uint32))
    
    dropout_threshold = 0.2
    dropout_mask = random_value > dropout_threshold
    dropout_mask_float = dropout_mask.to(tl.float32)
    
    group_norm_diff = input_value1 - input_value2
    group_norm_scaled = group_norm_diff * input_value3
    group_norm_weighted = group_norm_scaled * input_value4
    group_norm_result = group_norm_weighted + input_value5
    
    clamp_min = 0.0
    clamp_max = 1.0
    
    clamped_value = triton_helpers.minimum(group_norm_result, clamp_min)
    clamped_value = triton_helpers.maximum(clamped_value, clamp_min)
    clamped_value = triton_helpers.minimum(clamped_value, clamp_max)
    
    dropout_applied = dropout_mask_float * clamped_value
    dropout_scaled = dropout_applied * 1.25
    
    tl.store(output_ptr1 + (x0 + (12672 * x1)), dropout_mask, None)
    tl.store(output_ptr2 + (x4), dropout_scaled, None)