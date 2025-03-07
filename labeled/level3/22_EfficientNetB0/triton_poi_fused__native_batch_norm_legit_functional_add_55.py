# From: 22_EfficientNetB0

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_add_55poi_fused__native_batch_norm_legit_functional_add_55(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_bias, input_ptr_input, output_ptr, num_elements, BLOCK_SIZE: tl.constexpr
):
    num_elements = 219520
    block_offset = tl.program_id(0) * BLOCK_SIZE
    indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = indices < num_elements
    linear_index = indices
    channel_index = indices % 112

    input_data = tl.load(input_ptr_input + (linear_index), mask)
    mean = tl.load(input_ptr_mean + (channel_index), mask, eviction_policy='evict_last')
    variance = tl.load(input_ptr_var + (channel_index), mask, eviction_policy='evict_last')
    scale = tl.load(input_ptr_scale + (channel_index), mask, eviction_policy='evict_last')
    bias = tl.load(input_ptr_bias + (channel_index), mask, eviction_policy='evict_last')

    normalized_data = input_data - mean
    variance_normalized = variance / 1960.0
    epsilon = 1e-05
    variance_stabilized = variance_normalized + epsilon
    inv_std_dev = tl.extra.cuda.libdevice.rsqrt(variance_stabilized)

    scaled_data = normalized_data * inv_std_dev
    scaled_and_shifted = scaled_data * scale
    biased_data = scaled_and_shifted + bias
    output_data = biased_data + tl.load(input_ptr_input + (linear_index), mask)

    tl.store(output_ptr + (linear_index), output_data, mask)