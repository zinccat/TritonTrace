# From: 30_Gemm_GroupNorm_Hardtanh

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_hardtanh_native_group_norm_1poi_fused_hardtanh_native_group_norm_1(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, output_ptr0, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    index2 = index
    index0 = index % 512

    input_val0 = tl.load(input_ptr0 + index2, mask)
    input_val1 = tl.load(input_ptr1 + (index2 // 64), mask, eviction_policy='evict_last')
    input_val2 = tl.load(input_ptr2 + (index2 // 64), mask, eviction_policy='evict_last')
    input_val3 = tl.load(input_ptr3 + index0, mask, eviction_policy='evict_last')
    input_val4 = tl.load(input_ptr4 + index0, mask, eviction_policy='evict_last')

    subtracted = input_val0 - input_val1
    multiplied1 = subtracted * input_val2
    multiplied2 = multiplied1 * input_val3
    added = multiplied2 + input_val4

    lower_bound = -2.0
    upper_bound = 2.0

    clamped = triton_helpers.minimum(triton_helpers.maximum(added, lower_bound), upper_bound)

    tl.store(output_ptr0 + index2, clamped, mask)