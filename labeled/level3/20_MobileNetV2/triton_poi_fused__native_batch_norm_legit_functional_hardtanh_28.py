# From: 20_MobileNetV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_hardtanh_28poi_fused__native_batch_norm_legit_functional_hardtanh_28(input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_bias, input_ptr_input, output_ptr, total_elements, BLOCK_SIZE : tl.constexpr):
    total_elements = 1128960
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < total_elements
    global_indices = block_indices
    channel_indices = block_indices % 144

    input_data = tl.load(input_ptr_input + (global_indices), valid_mask)
    mean_data = tl.load(input_ptr_mean + (channel_indices), valid_mask, eviction_policy='evict_last')
    var_data = tl.load(input_ptr_var + (channel_indices), valid_mask, eviction_policy='evict_last')
    scale_data = tl.load(input_ptr_scale + (channel_indices), valid_mask, eviction_policy='evict_last')
    bias_data = tl.load(input_ptr_bias + (channel_indices), valid_mask, eviction_policy='evict_last')

    normalized_data = input_data - mean_data
    scaled_data = normalized_data * var_data
    scaled_and_shifted_data = scaled_data * scale_data
    batch_norm_output = scaled_and_shifted_data + bias_data

    clamped_output = triton_helpers.maximum(batch_norm_output, 0.0)
    hardtanh_output = triton_helpers.minimum(clamped_output, 6.0)

    tl.store(output_ptr + (global_indices), hardtanh_output, valid_mask)