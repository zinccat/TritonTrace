# From: 16_DenseNet201

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_171poi_fused__native_batch_norm_legit_functional_relu_171(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_shift, input_ptr_input, 
    output_ptr, num_elements, BLOCK_SIZE: tl.constexpr
):
    num_elements = 2132480
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = block_indices < num_elements
    global_indices = block_indices
    channel_indices = (block_indices // 196) % 1088

    mean = tl.load(input_ptr_mean + (global_indices), mask)
    variance = tl.load(input_ptr_var + (channel_indices), mask, eviction_policy='evict_last')
    scale = tl.load(input_ptr_scale + (channel_indices), mask, eviction_policy='evict_last')
    shift = tl.load(input_ptr_shift + (channel_indices), mask, eviction_policy='evict_last')
    input_data = tl.load(input_ptr_input + (global_indices), mask)

    normalized_data = input_data - mean
    variance_scale = 1960.0
    epsilon = 1e-05
    variance_adjusted = variance / variance_scale
    variance_adjusted_epsilon = variance_adjusted + epsilon
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_adjusted_epsilon)
    normalized_scaled = normalized_data * inv_sqrt_variance
    scaled_data = normalized_scaled * scale
    shifted_data = scaled_data + shift

    relu_output = tl.full([1], 0, tl.int32)
    relu_applied = triton_helpers.maximum(relu_output, shifted_data)
    tl.store(output_ptr + (global_indices), relu_applied, mask)