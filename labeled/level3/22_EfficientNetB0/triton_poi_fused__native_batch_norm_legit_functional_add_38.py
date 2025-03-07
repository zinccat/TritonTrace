# From: 22_EfficientNetB0

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_add_38poi_fused__native_batch_norm_legit_functional_add_38(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_bias, input_ptr_input, output_ptr, num_elements, BLOCK_SIZE: tl.constexpr
):
    num_elements = 313600
    block_offset = tl.program_id(0) * BLOCK_SIZE
    element_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = element_indices < num_elements
    global_indices = element_indices
    channel_indices = element_indices % 40

    mean = tl.load(input_ptr_mean + (global_indices), valid_mask)
    variance = tl.load(input_ptr_var + (channel_indices), valid_mask, eviction_policy='evict_last')
    scale = tl.load(input_ptr_scale + (channel_indices), valid_mask, eviction_policy='evict_last')
    bias = tl.load(input_ptr_bias + (channel_indices), valid_mask, eviction_policy='evict_last')
    input_data = tl.load(input_ptr_input + (global_indices), valid_mask)
    
    normalized_input = input_data - mean
    variance_adjusted = 7840.0
    epsilon = 1e-05
    variance_with_epsilon = variance / variance_adjusted + epsilon
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_with_epsilon)
    scaled_input = normalized_input * inv_sqrt_variance
    scaled_and_shifted = scaled_input * scale
    biased_output = scaled_and_shifted + bias
    final_output = biased_output + input_data

    tl.store(output_ptr + (global_indices), final_output, valid_mask)