# From: 19_MobileNetV1

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_6poi_fused__native_batch_norm_legit_functional_relu_6(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_bias, input_ptr_input, 
    output_ptr, num_elements, BLOCK_SIZE: tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    linear_index = block_indices
    element_index = block_indices % 32
    input_mean = tl.load(input_ptr_mean + (linear_index), None)
    input_var = tl.load(input_ptr_var + (element_index), None, eviction_policy='evict_last')
    input_scale = tl.load(input_ptr_scale + (element_index), None, eviction_policy='evict_last')
    input_bias = tl.load(input_ptr_bias + (element_index), None, eviction_policy='evict_last')
    input_data = tl.load(input_ptr_input + (linear_index), None)
    
    normalized_data = input_data - input_mean
    variance_scale = 125440.0
    normalized_variance = input_var / variance_scale
    epsilon = 1e-05
    adjusted_variance = normalized_variance + epsilon
    reciprocal_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(adjusted_variance)
    scaled_data = normalized_data * reciprocal_sqrt_variance
    scaled_and_shifted_data = scaled_data * input_scale
    output_data = scaled_and_shifted_data + input_bias
    
    zero_tensor = tl.full([1], 0, tl.int32)
    relu_output = triton_helpers.maximum(zero_tensor, output_data)
    tl.store(output_ptr + (linear_index), relu_output, None)