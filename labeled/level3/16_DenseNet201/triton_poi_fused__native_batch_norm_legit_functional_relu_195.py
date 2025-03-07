# From: 16_DenseNet201

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_195poi_fused__native_batch_norm_legit_functional_relu_195(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_shift, input_ptr_input, 
    output_ptr, total_elements, BLOCK_SIZE: tl.constexpr
):
    total_elements = 2885120
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < total_elements

    input_indices = block_indices
    channel_indices = (block_indices // 196) % 1472

    mean = tl.load(input_ptr_mean + (input_indices), valid_mask)
    variance = tl.load(input_ptr_var + (channel_indices), valid_mask, eviction_policy='evict_last')
    scale = tl.load(input_ptr_scale + (channel_indices), valid_mask, eviction_policy='evict_last')
    shift = tl.load(input_ptr_shift + (channel_indices), valid_mask, eviction_policy='evict_last')
    input_data = tl.load(input_ptr_input + (input_indices), valid_mask)

    normalized_data = input_data - mean
    variance_scale = 1960.0
    epsilon = 1e-05

    inv_stddev = tl.extra.cuda.libdevice.rsqrt(variance / variance_scale + epsilon)
    scaled_data = normalized_data * inv_stddev * scale
    output_data = scaled_data + shift

    zero_tensor = tl.full([1], 0, tl.int32)
    relu_output = triton_helpers.maximum(zero_tensor, output_data)

    tl.store(output_ptr + (input_indices), relu_output, valid_mask)