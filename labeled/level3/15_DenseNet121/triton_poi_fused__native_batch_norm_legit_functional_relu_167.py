# From: 15_DenseNet121

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_167poi_fused__native_batch_norm_legit_functional_relu_167(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_shift, input_ptr_input, 
    output_ptr, num_elements, BLOCK_SIZE: tl.constexpr
):
    num_elements = 344960
    block_offset = tl.program_id(0) * BLOCK_SIZE
    indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = indices < num_elements
    linear_index = indices
    channel_index = (indices // 49) % 704

    mean = tl.load(input_ptr_mean + (linear_index), mask)
    variance = tl.load(input_ptr_var + (channel_index), mask, eviction_policy='evict_last')
    scale = tl.load(input_ptr_scale + (channel_index), mask, eviction_policy='evict_last')
    shift = tl.load(input_ptr_shift + (channel_index), mask, eviction_policy='evict_last')
    input_data = tl.load(input_ptr_input + (linear_index), mask)

    normalized_input = input_data - mean
    variance_normalized = 490.0
    variance_adjusted = variance / variance_normalized
    epsilon = 1e-05
    variance_stabilized = variance_adjusted + epsilon
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_stabilized)

    scaled_input = normalized_input * inv_sqrt_variance
    scaled_and_shifted = scaled_input * scale
    output_data = scaled_and_shifted + shift

    zero_tensor = tl.full([1], 0, tl.int32)
    relu_output = triton_helpers.maximum(zero_tensor, output_data)

    tl.store(output_ptr + (linear_index), relu_output, mask)