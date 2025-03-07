# From: 15_DenseNet121

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_133poi_fused__native_batch_norm_legit_functional_relu_133(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_shift, input_ptr_input, 
    output_ptr, total_elements, BLOCK_SIZE: tl.constexpr
):
    total_elements = 1693440
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < total_elements
    input_indices = block_indices
    channel_indices = (block_indices // 196) % 864

    mean = tl.load(input_ptr_mean + (input_indices), valid_mask)
    variance = tl.load(input_ptr_var + (channel_indices), valid_mask, eviction_policy='evict_last')
    scale = tl.load(input_ptr_scale + (channel_indices), valid_mask, eviction_policy='evict_last')
    shift = tl.load(input_ptr_shift + (channel_indices), valid_mask, eviction_policy='evict_last')
    input_data = tl.load(input_ptr_input + (input_indices), valid_mask)

    normalized_data = input_data - mean
    variance_scale = 1960.0
    epsilon = 1e-05
    adjusted_variance = variance / variance_scale
    adjusted_variance += epsilon
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(adjusted_variance)
    normalized_scaled_data = normalized_data * inv_sqrt_variance
    scaled_data = normalized_scaled_data * scale
    shifted_data = scaled_data + shift

    relu_output = tl.full([1], 0, tl.int32)
    relu_applied = triton_helpers.maximum(relu_output, shifted_data)
    tl.store(output_ptr + (input_indices), relu_applied, valid_mask)