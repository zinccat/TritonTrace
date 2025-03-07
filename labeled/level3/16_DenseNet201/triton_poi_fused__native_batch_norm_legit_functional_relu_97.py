# From: 16_DenseNet201

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_97poi_fused__native_batch_norm_legit_functional_relu_97(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_shift, input_ptr_input, 
    output_ptr, num_elements, BLOCK_SIZE: tl.constexpr
):
    num_elements = 689920
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    input_indices = block_indices
    channel_indices = (block_indices // 196) % 352

    input_data = tl.load(input_ptr_input + (input_indices), valid_mask)
    mean_data = tl.load(input_ptr_mean + (channel_indices), valid_mask, eviction_policy='evict_last')
    var_data = tl.load(input_ptr_var + (channel_indices), valid_mask, eviction_policy='evict_last')
    scale_data = tl.load(input_ptr_scale + (channel_indices), valid_mask, eviction_policy='evict_last')
    shift_data = tl.load(input_ptr_shift + (channel_indices), valid_mask, eviction_policy='evict_last')

    normalized_data = input_data - mean_data
    variance_scale = 1960.0
    epsilon = 1e-05
    adjusted_variance = var_data / variance_scale
    adjusted_variance_with_epsilon = adjusted_variance + epsilon
    reciprocal_sqrt = tl.extra.cuda.libdevice.rsqrt(adjusted_variance_with_epsilon)
    scaled_normalized_data = normalized_data * reciprocal_sqrt
    scaled_data = scaled_normalized_data * scale_data
    shifted_data = scaled_data + shift_data

    zero_tensor = tl.full([1], 0, tl.int32)
    relu_output = triton_helpers.maximum(zero_tensor, shifted_data)
    tl.store(output_ptr + (input_indices), relu_output, valid_mask)