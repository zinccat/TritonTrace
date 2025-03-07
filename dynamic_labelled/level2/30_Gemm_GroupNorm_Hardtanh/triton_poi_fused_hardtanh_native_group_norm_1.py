# From: 30_Gemm_GroupNorm_Hardtanh

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_hardtanh_native_group_norm_1(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, 
    output_ptr0, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    element_index = index
    block_index = index % 512

    input_value0 = tl.load(input_ptr0 + (element_index), mask)
    input_value1 = tl.load(input_ptr1 + (element_index // 64), mask, eviction_policy='evict_last')
    input_value2 = tl.load(input_ptr2 + (element_index // 64), mask, eviction_policy='evict_last')
    input_value3 = tl.load(input_ptr3 + (block_index), mask, eviction_policy='evict_last')
    input_value4 = tl.load(input_ptr4 + (block_index), mask, eviction_policy='evict_last')

    normalized_value = input_value0 - input_value1
    scaled_value = normalized_value * input_value2
    weighted_value = scaled_value * input_value3
    biased_value = weighted_value + input_value4

    lower_bound = -2.0
    upper_bound = 2.0

    clamped_value = triton_helpers.minimum(triton_helpers.maximum(biased_value, lower_bound), upper_bound)

    tl.store(output_ptr0 + (element_index), clamped_value, mask)