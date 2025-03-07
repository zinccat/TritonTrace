# From: 20_MobileNetV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_hardtanh_25poi_fused__native_batch_norm_legit_functional_hardtanh_25(input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_shift, input_ptr_bias, output_ptr, total_elements, BLOCK_SIZE : tl.constexpr):
    total_elements = 4515840
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < total_elements
    global_indices = block_indices
    channel_indices = block_indices % 144

    input_mean = tl.load(input_ptr_mean + (global_indices), valid_mask)
    input_var = tl.load(input_ptr_var + (channel_indices), valid_mask, eviction_policy='evict_last')
    input_scale = tl.load(input_ptr_scale + (channel_indices), valid_mask, eviction_policy='evict_last')
    input_shift = tl.load(input_ptr_shift + (channel_indices), valid_mask, eviction_policy='evict_last')
    input_bias = tl.load(input_ptr_bias + (channel_indices), valid_mask, eviction_policy='evict_last')

    normalized_input = input_mean - input_var
    scaled_input = normalized_input * input_scale
    shifted_input = scaled_input * input_shift
    biased_input = shifted_input + input_bias

    clamped_min = 0.0
    clamped_max = 6.0

    clamped_output = triton_helpers.minimum(triton_helpers.maximum(biased_input, clamped_min), clamped_max)

    tl.store(output_ptr + (global_indices), clamped_output, valid_mask)