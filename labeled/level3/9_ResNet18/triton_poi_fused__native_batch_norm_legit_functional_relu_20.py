# From: 9_ResNet18

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_20poi_fused__native_batch_norm_legit_functional_relu_20(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_shift, input_ptr_input, 
    output_ptr, num_elements, BLOCK_SIZE: tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    global_indices = block_indices
    local_indices = (block_indices % 128)
    
    input_mean = tl.load(input_ptr_mean + (global_indices), None)
    input_var = tl.load(input_ptr_var + (local_indices), None, eviction_policy='evict_last')
    scale = tl.load(input_ptr_scale + (local_indices), None, eviction_policy='evict_last')
    shift = tl.load(input_ptr_shift + (local_indices), None, eviction_policy='evict_last')
    input_data = tl.load(input_ptr_input + (global_indices), None)
    
    centered_data = input_data - input_mean
    variance_scale = 1568.0
    normalized_variance = input_var / variance_scale
    epsilon = 1e-05
    adjusted_variance = normalized_variance + epsilon
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(adjusted_variance)
    normalized_data = centered_data * inv_sqrt_variance
    scaled_data = normalized_data * scale
    shifted_data = scaled_data + shift
    
    zero_tensor = tl.full([1], 0, tl.int32)
    relu_output = triton_helpers.maximum(zero_tensor, shifted_data)
    
    tl.store(output_ptr + (global_indices), relu_output, None)