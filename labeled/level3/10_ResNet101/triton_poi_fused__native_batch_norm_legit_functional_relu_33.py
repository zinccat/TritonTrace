# From: 10_ResNet101

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_33poi_fused__native_batch_norm_legit_functional_relu_33(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_shift, input_ptr_input, 
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
    input_shift = tl.load(input_ptr_shift + (local_indices), None, eviction_policy='evict_last')
    input_data = tl.load(input_ptr_input + (global_indices), None)
    
    normalized_data = input_data - input_mean
    variance_inverse = 7840.0
    epsilon = 1e-05
    normalized_variance = input_var / variance_inverse
    adjusted_variance = normalized_variance + epsilon
    rsqrt_variance = tl.extra.cuda.libdevice.rsqrt(adjusted_variance)
    scaled_data = normalized_data * rsqrt_variance
    scaled_and_shifted_data = scaled_data * input_scale
    final_output = scaled_and_shifted_data + input_shift
    
    relu_output = tl.full([1], 0, tl.int32)
    relu_applied_output = triton_helpers.maximum(relu_output, final_output)
    
    tl.store(output_ptr + (global_indices), relu_applied_output, None)