# From: 16_DenseNet201

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_201poi_fused__native_batch_norm_legit_functional_relu_201(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_shift, input_ptr_input, 
    output_ptr, num_elements, BLOCK_SIZE: tl.constexpr
):
    num_elements = 3073280
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    flat_index = block_indices
    channel_index = (block_indices // 196) % 1568

    mean = tl.load(input_ptr_mean + (flat_index), valid_mask)
    variance = tl.load(input_ptr_var + (channel_index), valid_mask, eviction_policy='evict_last')
    scale = tl.load(input_ptr_scale + (channel_index), valid_mask, eviction_policy='evict_last')
    shift = tl.load(input_ptr_shift + (channel_index), valid_mask, eviction_policy='evict_last')
    input_data = tl.load(input_ptr_input + (flat_index), valid_mask)

    normalized_input = input_data - mean
    variance_normalized = variance / 1960.0
    epsilon = 1e-05
    variance_stabilized = variance_normalized + epsilon
    inv_stddev = tl.extra.cuda.libdevice.rsqrt(variance_stabilized)

    scaled_input = normalized_input * inv_stddev
    scaled_and_shifted = scaled_input * scale + shift

    relu_output = tl.full([1], 0, tl.int32)
    relu_applied = triton_helpers.maximum(relu_output, scaled_and_shifted)

    tl.store(output_ptr + (flat_index), relu_applied, valid_mask)