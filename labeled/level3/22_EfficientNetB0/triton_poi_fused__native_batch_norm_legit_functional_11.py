# From: 22_EfficientNetB0

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_11poi_fused__native_batch_norm_legit_functional_11(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_bias, input_ptr_input, 
    output_ptr, num_elements, BLOCK_SIZE : tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    input_indices = block_indices
    element_indices = input_indices % 16
    
    mean_value = tl.load(input_ptr_mean + (input_indices), None)
    variance_value = tl.load(input_ptr_var + (element_indices), None, eviction_policy='evict_last')
    scale_value = tl.load(input_ptr_scale + (element_indices), None, eviction_policy='evict_last')
    bias_value = tl.load(input_ptr_bias + (element_indices), None, eviction_policy='evict_last')
    input_value = tl.load(input_ptr_input + (input_indices), None)
    
    normalized_input = input_value - mean_value
    variance_adjusted = 125440.0
    variance_normalized = variance_value / variance_adjusted
    epsilon = 1e-05
    variance_stabilized = variance_normalized + epsilon
    reciprocal_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_stabilized)
    scaled_normalized_input = normalized_input * reciprocal_sqrt_variance
    scaled_input = scaled_normalized_input * scale_value
    output_value = scaled_input + bias_value
    
    tl.store(output_ptr + (input_indices), output_value, None)