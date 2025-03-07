# From: 16_DenseNet201

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_169poi_fused__native_batch_norm_legit_functional_relu_169(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_shift, input_ptr_input, 
    output_ptr, num_elements, BLOCK_SIZE: tl.constexpr
):
    num_elements = 2069760
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    global_indices = block_indices
    channel_indices = (block_indices // 196) % 1056

    mean = tl.load(input_ptr_mean + (global_indices), valid_mask)
    variance = tl.load(input_ptr_var + (channel_indices), valid_mask, eviction_policy='evict_last')
    scale = tl.load(input_ptr_scale + (channel_indices), valid_mask, eviction_policy='evict_last')
    shift = tl.load(input_ptr_shift + (channel_indices), valid_mask, eviction_policy='evict_last')
    input_data = tl.load(input_ptr_input + (global_indices), valid_mask)

    normalized_data = input_data - mean
    variance_normalized = variance / 1960.0
    epsilon = 1e-05
    variance_stabilized = variance_normalized + epsilon
    inv_stddev = tl.extra.cuda.libdevice.rsqrt(variance_stabilized)
    scaled_data = normalized_data * inv_stddev
    scaled_and_shifted_data = scaled_data * scale + shift

    relu_output = tl.full([1], 0, tl.int32)
    relu_applied = triton_helpers.maximum(relu_output, scaled_and_shifted_data)
    tl.store(output_ptr + (global_indices), relu_applied, valid_mask)