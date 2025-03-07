# From: 22_EfficientNetB0

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_hardtanh_19poi_fused__native_batch_norm_legit_functional_hardtanh_19(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_shift, input_ptr_out, output_ptr, num_elements, BLOCK_SIZE: tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    global_indices = block_indices
    local_indices = block_indices % 96

    mean_value = tl.load(input_ptr_mean + (global_indices), None)
    variance_value = tl.load(input_ptr_var + (local_indices), None, eviction_policy='evict_last')
    scale_value = tl.load(input_ptr_scale + (local_indices), None, eviction_policy='evict_last')
    shift_value = tl.load(input_ptr_shift + (local_indices), None, eviction_policy='evict_last')
    output_value = tl.load(input_ptr_out + (local_indices), None, eviction_policy='evict_last')

    normalized_value = (mean_value - variance_value) * variance_value * scale_value
    adjusted_value = normalized_value + shift_value

    clamped_value = triton_helpers.maximum(adjusted_value, 0.0)
    final_output = triton_helpers.minimum(clamped_value, 6.0)

    tl.store(output_ptr + (global_indices), final_output, None)