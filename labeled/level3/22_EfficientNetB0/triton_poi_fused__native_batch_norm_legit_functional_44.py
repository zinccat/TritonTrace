# From: 22_EfficientNetB0

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_44poi_fused__native_batch_norm_legit_functional_44(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_bias, input_ptr_input, 
    output_ptr, num_elements, BLOCK_SIZE : tl.constexpr
):
    num_elements = 156800
    block_offset = tl.program_id(0) * BLOCK_SIZE
    indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = indices < num_elements
    linear_index = indices
    channel_index = indices % 80

    mean = tl.load(input_ptr_mean + (linear_index), mask)
    variance = tl.load(input_ptr_var + (channel_index), mask, eviction_policy='evict_last')
    scale = tl.load(input_ptr_scale + (channel_index), mask, eviction_policy='evict_last')
    bias = tl.load(input_ptr_bias + (channel_index), mask, eviction_policy='evict_last')
    input_data = tl.load(input_ptr_input + (linear_index), mask)

    centered_input = input_data - mean
    variance_normalized = variance / 1960.0
    epsilon = 1e-05
    variance_stabilized = variance_normalized + epsilon
    inv_stddev = tl.extra.cuda.libdevice.rsqrt(variance_stabilized)

    normalized_input = centered_input * inv_stddev
    scaled_input = normalized_input * scale
    output_data = scaled_input + bias

    tl.store(output_ptr + (linear_index), output_data, mask)