# From: 15_DenseNet121

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_161poi_fused__native_batch_norm_legit_functional_relu_161(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_shift, input_ptr_input, 
    output_ptr, num_elements, BLOCK_SIZE: tl.constexpr
):
    num_elements = 313600
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = block_indices < num_elements
    linear_index = block_indices
    channel_index = (block_indices // 49) % 640

    input_data = tl.load(input_ptr_input + (linear_index), mask)
    mean = tl.load(input_ptr_mean + (channel_index), mask, eviction_policy='evict_last')
    variance = tl.load(input_ptr_var + (channel_index), mask, eviction_policy='evict_last')
    scale = tl.load(input_ptr_scale + (channel_index), mask, eviction_policy='evict_last')
    shift = tl.load(input_ptr_shift + (channel_index), mask, eviction_policy='evict_last')

    normalized_data = input_data - mean
    variance_scale = 490.0
    normalized_variance = variance / variance_scale
    epsilon = 1e-05
    adjusted_variance = normalized_variance + epsilon
    inv_std_dev = tl.extra.cuda.libdevice.rsqrt(adjusted_variance)
    scaled_data = normalized_data * inv_std_dev
    scaled_and_shifted_data = scaled_data * scale
    output_data = scaled_and_shifted_data + shift

    zero_tensor = tl.full([1], 0, tl.int32)
    relu_output = triton_helpers.maximum(zero_tensor, output_data)
    tl.store(output_ptr + (linear_index), relu_output, mask)