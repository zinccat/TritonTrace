# From: 30_Gemm_GroupNorm_Hardtanh

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused_hardtanh_native_group_norm_1(input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, output_ptr0, xnumel, XBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    x_index_2 = x_index
    x_index_0 = x_index % 512
    
    input_value0 = tl.load(input_ptr0 + (x_index_2), None)
    input_value1 = tl.load(input_ptr1 + ((x_index_2 // 64)), None, eviction_policy='evict_last')
    input_value2 = tl.load(input_ptr2 + ((x_index_2 // 64)), None, eviction_policy='evict_last')
    input_value3 = tl.load(input_ptr3 + (x_index_0), None, eviction_policy='evict_last')
    input_value4 = tl.load(input_ptr4 + (x_index_0), None, eviction_policy='evict_last')
    
    normalized_value = input_value0 - input_value1
    scaled_value = normalized_value * input_value2
    weighted_value = scaled_value * input_value3
    adjusted_value = weighted_value + input_value4
    
    lower_bound = -2.0
    upper_bound = 2.0
    
    clamped_value = triton_helpers.minimum(triton_helpers.maximum(adjusted_value, lower_bound), upper_bound)
    
    tl.store(output_ptr0 + (x_index_2), clamped_value, None)