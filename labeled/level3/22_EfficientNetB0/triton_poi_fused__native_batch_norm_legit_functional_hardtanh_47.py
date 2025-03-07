# From: 22_EfficientNetB0

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_hardtanh_47poi_fused__native_batch_norm_legit_functional_hardtanh_47(input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_shift, input_ptr_out, output_ptr, num_elements, BLOCK_SIZE : tl.constexpr):
    num_elements = 940800
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    global_indices = block_indices
    channel_indices = block_indices % 480

    input_mean = tl.load(input_ptr_mean + (global_indices), valid_mask)
    input_var = tl.load(input_ptr_var + (channel_indices), valid_mask, eviction_policy='evict_last')
    input_scale = tl.load(input_ptr_scale + (channel_indices), valid_mask, eviction_policy='evict_last')
    input_shift = tl.load(input_ptr_shift + (channel_indices), valid_mask, eviction_policy='evict_last')
    output_ptr_base = tl.load(input_ptr_out + (channel_indices), valid_mask, eviction_policy='evict_last')

    normalized_input = input_mean - input_var
    scaled_input = normalized_input * input_scale
    scaled_shifted_input = scaled_input * input_shift
    final_output = scaled_shifted_input + output_ptr_base

    clamped_output = triton_helpers.maximum(final_output, 0.0)
    clamped_output = triton_helpers.minimum(clamped_output, 6.0)

    tl.store(output_ptr + (global_indices), clamped_output, valid_mask)