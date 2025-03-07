# From: 15_DenseNet121

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_176poi_fused__native_batch_norm_legit_functional_relu_176(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_shift, input_ptr_input, 
    output_ptr, num_elements, BLOCK_SIZE: tl.constexpr
):
    num_elements = 392000
    block_offset = tl.program_id(0) * BLOCK_SIZE
    indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = indices < num_elements
    flat_indices = indices
    channel_indices = (indices // 49) % 800

    input_data = tl.load(input_ptr_input + (flat_indices), mask)
    mean = tl.load(input_ptr_mean + (channel_indices), mask, eviction_policy='evict_last')
    variance = tl.load(input_ptr_var + (channel_indices), mask, eviction_policy='evict_last')
    scale = tl.load(input_ptr_scale + (channel_indices), mask, eviction_policy='evict_last')
    shift = tl.load(input_ptr_shift + (channel_indices), mask, eviction_policy='evict_last')

    normalized_data = input_data - mean
    variance_scale = 490.0
    variance_adjusted = variance / variance_scale
    epsilon = 1e-05
    variance_stabilized = variance_adjusted + epsilon
    inv_stddev = tl.extra.cuda.libdevice.rsqrt(variance_stabilized)

    scaled_data = normalized_data * inv_stddev
    scaled_and_shifted = scaled_data * scale
    output_data = scaled_and_shifted + shift

    zero_tensor = tl.full([1], 0, tl.int32)
    relu_output = triton_helpers.maximum(zero_tensor, output_data)

    tl.store(output_ptr + (flat_indices), relu_output, mask)