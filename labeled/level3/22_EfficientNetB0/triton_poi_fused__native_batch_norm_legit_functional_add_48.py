# From: 22_EfficientNetB0

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_add_48poi_fused__native_batch_norm_legit_functional_add_48(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_bias, input_ptr_input, output_ptr, num_elements, BLOCK_SIZE: tl.constexpr
):
    num_elements = 156800
    block_offset = tl.program_id(0) * BLOCK_SIZE
    element_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = element_indices < num_elements
    global_indices = element_indices
    channel_indices = element_indices % 80

    input_data = tl.load(input_ptr_input + (global_indices), valid_mask)
    mean_data = tl.load(input_ptr_mean + (channel_indices), valid_mask, eviction_policy='evict_last')
    var_data = tl.load(input_ptr_var + (channel_indices), valid_mask, eviction_policy='evict_last')
    scale_data = tl.load(input_ptr_scale + (channel_indices), valid_mask, eviction_policy='evict_last')
    bias_data = tl.load(input_ptr_bias + (channel_indices), valid_mask, eviction_policy='evict_last')
    output_data = tl.load(output_ptr + (global_indices), valid_mask)

    normalized_data = input_data - mean_data
    variance_epsilon = 1e-05
    normalized_variance = var_data / 1960.0 + variance_epsilon
    inv_std_dev = tl.extra.cuda.libdevice.rsqrt(normalized_variance)
    scaled_data = normalized_data * inv_std_dev
    scaled_and_shifted_data = scaled_data * scale_data + bias_data
    final_output = scaled_and_shifted_data + output_data

    tl.store(output_ptr + (global_indices), final_output, valid_mask)