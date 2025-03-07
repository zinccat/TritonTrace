# From: 20_MobileNetV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_21poi_fused__native_batch_norm_legit_functional_21(input_ptr_mean, input_ptr_var, input_ptr_beta, input_ptr_gamma, input_ptr_bias, output_ptr, total_elements, BLOCK_SIZE : tl.constexpr):
    total_elements = 752640
    block_offset = tl.program_id(0) * BLOCK_SIZE
    element_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = element_indices < total_elements
    linear_index = element_indices
    channel_index = element_indices % 24
    
    input_mean = tl.load(input_ptr_mean + (linear_index), valid_mask)
    input_var = tl.load(input_ptr_var + (channel_index), valid_mask, eviction_policy='evict_last')
    beta = tl.load(input_ptr_beta + (channel_index), valid_mask, eviction_policy='evict_last')
    gamma = tl.load(input_ptr_gamma + (channel_index), valid_mask, eviction_policy='evict_last')
    bias = tl.load(input_ptr_bias + (channel_index), valid_mask, eviction_policy='evict_last')
    
    normalized_input = input_mean - beta
    variance_epsilon = 1e-05
    normalized_variance = input_var / 31360.0
    adjusted_variance = normalized_variance + variance_epsilon
    inv_std_dev = tl.extra.cuda.libdevice.rsqrt(adjusted_variance)
    scaled_input = normalized_input * inv_std_dev
    scaled_and_shifted = scaled_input * gamma
    output = scaled_and_shifted + bias
    
    tl.store(output_ptr + (linear_index), output, valid_mask)