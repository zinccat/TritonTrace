# From: 21_EfficientNetMBConv

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_hardtanh_7poi_fused__native_batch_norm_legit_functional_hardtanh_7(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, output_ptr0, num_elements, BLOCK_SIZE : tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    global_indices = block_indices
    local_indices = block_indices % 672

    input_value0 = tl.load(input_ptr0 + (global_indices), None)
    input_value1 = tl.load(input_ptr1 + (local_indices), None, eviction_policy='evict_last')
    input_value2 = tl.load(input_ptr2 + (local_indices), None, eviction_policy='evict_last')
    input_value3 = tl.load(input_ptr3 + (local_indices), None, eviction_policy='evict_last')
    input_value4 = tl.load(input_ptr4 + (local_indices), None, eviction_policy='evict_last')

    normalized_value = (input_value0 - input_value1) * input_value2 * input_value3 + input_value4
    clamped_value = triton_helpers.maximum(normalized_value, 0.0)
    output_value = triton_helpers.minimum(clamped_value, 6.0)

    tl.store(output_ptr0 + (global_indices), output_value, None)