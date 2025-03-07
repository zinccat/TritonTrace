# From: 15_DenseNet121

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_146poi_fused__native_batch_norm_legit_functional_relu_146(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_shift, input_ptr_input, 
    output_ptr, num_elements, BLOCK_SIZE: tl.constexpr
):
    num_elements = 250880
    block_offset = tl.program_id(0) * BLOCK_SIZE
    indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = indices < num_elements
    linear_index = indices
    channel_index = (indices // 49) % 512

    input_data = tl.load(input_ptr_input + (linear_index), mask)
    mean = tl.load(input_ptr_mean + (channel_index), mask, eviction_policy='evict_last')
    variance = tl.load(input_ptr_var + (channel_index), mask, eviction_policy='evict_last')
    scale = tl.load(input_ptr_scale + (channel_index), mask, eviction_policy='evict_last')
    shift = tl.load(input_ptr_shift + (channel_index), mask, eviction_policy='evict_last')

    normalized_data = input_data - mean
    variance_normalized = variance / 490.0
    epsilon = 1e-05
    variance_stabilized = variance_normalized + epsilon
    inv_stddev = tl.extra.cuda.libdevice.rsqrt(variance_stabilized)
    scaled_data = normalized_data * inv_stddev
    scaled_and_shifted_data = scaled_data * scale + shift

    output_data = tl.full([1], 0, tl.int32)
    relu_output = triton_helpers.maximum(output_data, scaled_and_shifted_data)
    tl.store(output_ptr + (linear_index), relu_output, mask)