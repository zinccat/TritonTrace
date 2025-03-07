# From: 19_MobileNetV1

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_24poi_fused__native_batch_norm_legit_functional_relu_24(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_bias, input_ptr_input, 
    output_ptr, num_elements, BLOCK_SIZE: tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    global_indices = block_indices
    local_indices = (block_indices % 256)
    
    input_mean = tl.load(input_ptr_mean + (global_indices), None)
    input_var = tl.load(input_ptr_var + (local_indices), None, eviction_policy='evict_last')
    input_scale = tl.load(input_ptr_scale + (local_indices), None, eviction_policy='evict_last')
    input_bias = tl.load(input_ptr_bias + (local_indices), None, eviction_policy='evict_last')
    input_data = tl.load(input_ptr_input + (global_indices), None)
    
    normalized_data = input_data - input_mean
    variance_inverse_sqrt = 7840.0
    epsilon = 1e-05
    adjusted_variance = input_var / variance_inverse_sqrt
    variance_with_epsilon = adjusted_variance + epsilon
    rsqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_with_epsilon)
    scaled_normalized_data = normalized_data * rsqrt_variance
    scaled_data = scaled_normalized_data * input_scale
    biased_data = scaled_data + input_bias
    
    zero_tensor = tl.full([1], 0, tl.int32)
    relu_output = triton_helpers.maximum(zero_tensor, biased_data)
    
    tl.store(output_ptr + (global_indices), relu_output, None)