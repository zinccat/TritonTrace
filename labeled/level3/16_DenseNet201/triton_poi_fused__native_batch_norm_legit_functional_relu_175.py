# From: 16_DenseNet201

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_175poi_fused__native_batch_norm_legit_functional_relu_175(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_shift, input_ptr_input, 
    output_ptr, total_elements, BLOCK_SIZE: tl.constexpr
):
    total_elements = 2257920
    block_offset = tl.program_id(0) * BLOCK_SIZE
    element_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = element_indices < total_elements
    flat_indices = element_indices
    channel_indices = (element_indices // 196) % 1152

    input_data = tl.load(input_ptr_input + (flat_indices), valid_mask)
    mean_data = tl.load(input_ptr_mean + (channel_indices), valid_mask, eviction_policy='evict_last')
    var_data = tl.load(input_ptr_var + (channel_indices), valid_mask, eviction_policy='evict_last')
    scale_data = tl.load(input_ptr_scale + (channel_indices), valid_mask, eviction_policy='evict_last')
    shift_data = tl.load(input_ptr_shift + (channel_indices), valid_mask, eviction_policy='evict_last')

    normalized_data = input_data - mean_data
    variance_scale = 1960.0
    epsilon = 1e-05
    adjusted_variance = var_data / variance_scale
    variance_with_epsilon = adjusted_variance + epsilon
    reciprocal_sqrt = tl.extra.cuda.libdevice.rsqrt(variance_with_epsilon)
    scaled_data = normalized_data * reciprocal_sqrt
    scaled_and_shifted_data = scaled_data * scale_data
    final_output = scaled_and_shifted_data + shift_data

    zero_tensor = tl.full([1], 0, tl.int32)
    relu_output = triton_helpers.maximum(zero_tensor, final_output)
    tl.store(output_ptr + (flat_indices), relu_output, valid_mask)