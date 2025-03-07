# From: 19_MobileNetV1

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_33poi_fused__native_batch_norm_legit_functional_relu_33(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_bias, input_ptr_input, 
    output_ptr, num_elements, BLOCK_SIZE: tl.constexpr
):
    num_elements = 250880
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    global_indices = block_indices
    channel_indices = block_indices % 512

    input_mean = tl.load(input_ptr_mean + (global_indices), valid_mask)
    input_var = tl.load(input_ptr_var + (channel_indices), valid_mask, eviction_policy='evict_last')
    scale = tl.load(input_ptr_scale + (channel_indices), valid_mask, eviction_policy='evict_last')
    bias = tl.load(input_ptr_bias + (channel_indices), valid_mask, eviction_policy='evict_last')
    input_data = tl.load(input_ptr_input + (global_indices), valid_mask)

    normalized_input = input_data - input_mean
    variance_inverse_sqrt = tl.full([1], 490.0, tl.float32)
    epsilon = tl.full([1], 1e-05, tl.float32)
    normalized_variance = input_var / variance_inverse_sqrt
    normalized_variance_with_epsilon = normalized_variance + epsilon
    rsqrt_normalized_variance = tl.extra.cuda.libdevice.rsqrt(normalized_variance_with_epsilon)

    scaled_normalized_input = normalized_input * rsqrt_normalized_variance
    scaled_input = scaled_normalized_input * scale
    biased_input = scaled_input + bias

    relu_output = tl.full([1], 0, tl.int32)
    relu_applied_output = triton_helpers.maximum(relu_output, biased_input)

    tl.store(output_ptr + (global_indices), relu_applied_output, valid_mask)