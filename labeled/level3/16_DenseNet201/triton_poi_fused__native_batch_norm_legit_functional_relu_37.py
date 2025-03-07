# From: 16_DenseNet201

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_37poi_fused__native_batch_norm_legit_functional_relu_37(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_shift, input_ptr_input, 
    output_ptr, num_elements, BLOCK_SIZE: tl.constexpr
):
    num_elements = 1254400
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = block_indices < num_elements
    flat_indices = block_indices
    batch_indices = (block_indices // 784) % 160

    input_data = tl.load(input_ptr_input + (flat_indices), mask)
    mean = tl.load(input_ptr_mean + (batch_indices), mask, eviction_policy='evict_last')
    variance = tl.load(input_ptr_var + (batch_indices), mask, eviction_policy='evict_last')
    scale = tl.load(input_ptr_scale + (batch_indices), mask, eviction_policy='evict_last')
    shift = tl.load(input_ptr_shift + (batch_indices), mask, eviction_policy='evict_last')

    normalized_data = input_data - mean
    variance_factor = 7840.0
    adjusted_variance = variance / variance_factor
    epsilon = 1e-05
    variance_stabilizer = adjusted_variance + epsilon
    inv_std_dev = tl.extra.cuda.libdevice.rsqrt(variance_stabilizer)

    scaled_data = normalized_data * inv_std_dev
    scaled_shifted_data = scaled_data * scale
    output_data = scaled_shifted_data + shift

    zero_tensor = tl.full([1], 0, tl.int32)
    relu_output = triton_helpers.maximum(zero_tensor, output_data)

    tl.store(output_ptr + (flat_indices), relu_output, mask)