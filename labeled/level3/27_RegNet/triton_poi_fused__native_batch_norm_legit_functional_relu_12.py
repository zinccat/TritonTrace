# From: 27_RegNet

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_12poi_fused__native_batch_norm_legit_functional_relu_12(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_shift, input_ptr_input, 
    output_ptr, num_elements, BLOCK_SIZE : tl.constexpr):

    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    linear_indices = block_indices
    element_indices = (block_indices % 64)
    
    input_mean = tl.load(input_ptr_mean + (linear_indices), None)
    input_var = tl.load(input_ptr_var + (element_indices), None, eviction_policy='evict_last')
    scale = tl.load(input_ptr_scale + (element_indices), None, eviction_policy='evict_last')
    shift = tl.load(input_ptr_shift + (element_indices), None, eviction_policy='evict_last')
    input_data = tl.load(input_ptr_input + (element_indices), None, eviction_policy='evict_last')
    
    normalized_data = input_data - input_mean
    variance_factor = 401408.0
    normalized_variance = input_var / variance_factor
    epsilon = 1e-05
    adjusted_variance = normalized_variance + epsilon
    reciprocal_sqrt = tl.extra.cuda.libdevice.rsqrt(adjusted_variance)
    scaled_data = normalized_data * reciprocal_sqrt
    scaled_and_shifted_data = scaled_data * scale
    final_output = scaled_and_shifted_data + shift
    
    zero_tensor = tl.full([1], 0, tl.int32)
    relu_output = triton_helpers.maximum(zero_tensor, final_output)
    
    tl.store(output_ptr + (linear_indices), relu_output, None)