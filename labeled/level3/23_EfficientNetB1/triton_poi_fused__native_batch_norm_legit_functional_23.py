# From: 23_EfficientNetB1

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_23poi_fused__native_batch_norm_legit_functional_23(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_bias, input_ptr_input, 
    output_ptr, total_elements, BLOCK_SIZE : tl.constexpr
):
    total_elements = 864000
    block_offset = tl.program_id(0) * BLOCK_SIZE
    element_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = element_indices < total_elements
    global_indices = element_indices
    channel_indices = element_indices % 24

    mean = tl.load(input_ptr_mean + (global_indices), valid_mask)
    variance = tl.load(input_ptr_var + (channel_indices), valid_mask, eviction_policy='evict_last')
    scale = tl.load(input_ptr_scale + (channel_indices), valid_mask, eviction_policy='evict_last')
    bias = tl.load(input_ptr_bias + (channel_indices), valid_mask, eviction_policy='evict_last')
    input_data = tl.load(input_ptr_input + (global_indices), valid_mask)

    normalized_data = input_data - mean
    variance_normalized = 36000.0
    variance_adjusted = variance / variance_normalized
    epsilon = 1e-05
    variance_stabilized = variance_adjusted + epsilon
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_stabilized)
    scaled_data = normalized_data * inv_sqrt_variance
    scaled_and_shifted = scaled_data * scale
    output_data = scaled_and_shifted + bias

    tl.store(output_ptr + (global_indices), output_data, valid_mask)