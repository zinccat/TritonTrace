# From: 16_DenseNet201

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_211poi_fused__native_batch_norm_legit_functional_relu_211(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_shift, input_ptr_input, 
    output_ptr, num_elements, BLOCK_SIZE: tl.constexpr
):
    num_elements = 3386880
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = block_indices < num_elements
    linear_index = block_indices
    channel_index = (block_indices // 196) % 1728

    mean = tl.load(input_ptr_mean + (linear_index), mask)
    variance = tl.load(input_ptr_var + (channel_index), mask, eviction_policy='evict_last')
    scale = tl.load(input_ptr_scale + (channel_index), mask, eviction_policy='evict_last')
    shift = tl.load(input_ptr_shift + (channel_index), mask, eviction_policy='evict_last')
    input_data = tl.load(input_ptr_input + (linear_index), mask)

    normalized_data = input_data - mean
    variance_epsilon = 1e-05
    normalized_variance = variance / 1960.0 + variance_epsilon
    inv_stddev = tl.extra.cuda.libdevice.rsqrt(normalized_variance)
    scaled_data = normalized_data * inv_stddev * scale
    output_data = scaled_data + shift

    zero_tensor = tl.full([1], 0, tl.int32)
    relu_output = triton_helpers.maximum(zero_tensor, output_data)
    tl.store(output_ptr + (linear_index), relu_output, mask)