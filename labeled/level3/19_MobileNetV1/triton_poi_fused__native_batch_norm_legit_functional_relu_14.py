# From: 19_MobileNetV1

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_14poi_fused__native_batch_norm_legit_functional_relu_14(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_bias, input_ptr_input, 
    output_ptr, num_elements, BLOCK_SIZE: tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    linear_index = block_indices
    element_index = block_indices % 64
    input_mean = tl.load(input_ptr_mean + (linear_index), None)
    input_var = tl.load(input_ptr_var + (element_index), None, eviction_policy='evict_last')
    scale = tl.load(input_ptr_scale + (element_index), None, eviction_policy='evict_last')
    bias = tl.load(input_ptr_bias + (element_index), None, eviction_policy='evict_last')
    input_data = tl.load(input_ptr_input + (linear_index), None)
    
    normalized_data = input_data - input_mean
    variance_inverse_sqrt = 31360.0
    epsilon = 1e-05
    adjusted_variance = scale / variance_inverse_sqrt + epsilon
    variance_rsqrt = tl.extra.cuda.libdevice.rsqrt(adjusted_variance)
    normalized_scaled_data = normalized_data * variance_rsqrt
    scaled_data = normalized_scaled_data * scale
    biased_data = scaled_data + bias
    
    zero_tensor = tl.full([1], 0, tl.int32)
    relu_output = triton_helpers.maximum(zero_tensor, biased_data)
    
    tl.store(output_ptr + (linear_index), relu_output, None)