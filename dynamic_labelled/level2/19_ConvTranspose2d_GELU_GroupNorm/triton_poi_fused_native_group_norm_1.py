# From: 19_ConvTranspose2d_GELU_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_group_norm_1poi_fused_native_group_norm_1(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_bias, output_ptr, num_elements, BLOCK_SIZE: tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    linear_index = block_indices
    group_index = linear_index // 4356
    channel_index = (linear_index // 4356) % 64

    mean = tl.load(input_ptr_mean + (linear_index), valid_mask)
    variance = tl.load(input_ptr_var + (group_index // 8), valid_mask, eviction_policy='evict_last')
    scale = tl.load(input_ptr_scale + (group_index // 8), valid_mask, eviction_policy='evict_last')
    bias = tl.load(input_ptr_bias + (channel_index), valid_mask, eviction_policy='evict_last')
    output_bias = tl.load(input_ptr_bias + (channel_index), valid_mask, eviction_policy='evict_last')

    half = 0.5
    scaled_mean = mean * half
    sqrt_inv_two = 0.7071067811865476
    erf_input = mean * sqrt_inv_two
    erf_result = tl.extra.cuda.libdevice.erf(erf_input)
    one = 1.0
    erf_adjusted = erf_result + one
    gelu_output = scaled_mean * erf_adjusted
    normalized_output = gelu_output - variance

    variance_scale = 34848.0
    variance_adjusted = variance / variance_scale
    epsilon = 1e-05
    variance_with_epsilon = variance_adjusted + epsilon
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_with_epsilon)
    normalized_scaled_output = normalized_output * inv_sqrt_variance
    scaled_output = normalized_scaled_output * scale
    final_output = scaled_output + bias

    tl.store(output_ptr + (linear_index), final_output, valid_mask)