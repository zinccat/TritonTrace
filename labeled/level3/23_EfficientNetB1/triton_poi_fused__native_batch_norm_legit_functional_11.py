# From: 23_EfficientNetB1

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_11poi_fused__native_batch_norm_legit_functional_11(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_bias, input_ptr_input, 
    output_ptr, total_elements, BLOCK_SIZE : tl.constexpr
):
    total_elements = 2304000
    block_offset = tl.program_id(0) * BLOCK_SIZE
    element_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = element_indices < total_elements
    linear_index = element_indices
    channel_index = element_indices % 16
    
    mean = tl.load(input_ptr_mean + (linear_index), valid_mask)
    variance = tl.load(input_ptr_var + (channel_index), valid_mask, eviction_policy='evict_last')
    scale = tl.load(input_ptr_scale + (channel_index), valid_mask, eviction_policy='evict_last')
    bias = tl.load(input_ptr_bias + (channel_index), valid_mask, eviction_policy='evict_last')
    input_data = tl.load(input_ptr_input + (linear_index), valid_mask)
    
    normalized_input = input_data - mean
    variance_normalized = 144000.0
    epsilon = 1e-05
    variance_adjusted = variance / variance_normalized
    variance_adjusted_epsilon = variance_adjusted + epsilon
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_adjusted_epsilon)
    scaled_normalized_input = normalized_input * inv_sqrt_variance
    scaled_input = scaled_normalized_input * scale
    output_data = scaled_input + bias
    
    tl.store(output_ptr + (linear_index), output_data, valid_mask)