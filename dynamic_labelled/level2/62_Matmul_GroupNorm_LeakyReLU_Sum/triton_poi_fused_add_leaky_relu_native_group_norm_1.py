# From: 62_Matmul_GroupNorm_LeakyReLU_Sum

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_leaky_relu_native_group_norm_1poi_fused_add_leaky_relu_native_group_norm_1(
    in_out_ptr, input_ptr, mean_ptr, variance_ptr, gamma_ptr, beta_ptr, num_elements, BLOCK_SIZE: tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    global_indices = block_indices
    channel_indices = block_indices % 256

    input_data = tl.load(input_ptr + (global_indices), valid_mask)
    mean_data = tl.load(mean_ptr + (global_indices // 32), valid_mask, eviction_policy='evict_last')
    variance_data = tl.load(variance_ptr + (global_indices // 32), valid_mask, eviction_policy='evict_last')
    gamma_data = tl.load(gamma_ptr + (channel_indices), valid_mask, eviction_policy='evict_last')
    beta_data = tl.load(beta_ptr + (channel_indices), valid_mask, eviction_policy='evict_last')

    normalized_data = input_data - mean_data
    scaled_variance = normalized_data * variance_data
    scaled_data = scaled_variance * gamma_data
    shifted_data = scaled_data + beta_data

    leaky_relu_threshold = 0.0
    leaky_relu_slope = 0.01

    positive_mask = shifted_data > leaky_relu_threshold
    leaky_data = shifted_data * leaky_relu_slope
    activated_data = tl.where(positive_mask, shifted_data, leaky_data)
    output_data = activated_data + activated_data

    tl.store(in_out_ptr + (global_indices), output_data, valid_mask)