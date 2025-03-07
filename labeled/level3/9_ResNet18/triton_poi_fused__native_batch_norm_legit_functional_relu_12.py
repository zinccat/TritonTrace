# From: 9_ResNet18

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_12poi_fused__native_batch_norm_legit_functional_relu_12(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_shift, input_ptr_input, 
    output_ptr, num_elements, BLOCK_SIZE: tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    element_indices = block_indices
    element_index_mod = (block_indices % 64)
    
    input_mean = tl.load(input_ptr_mean + (element_indices), None)
    input_var = tl.load(input_ptr_var + (element_index_mod), None, eviction_policy='evict_last')
    input_scale = tl.load(input_ptr_scale + (element_index_mod), None, eviction_policy='evict_last')
    input_shift = tl.load(input_ptr_shift + (element_index_mod), None, eviction_policy='evict_last')
    input_data = tl.load(input_ptr_input + (element_indices), None)
    
    normalized_data = input_data - input_mean
    variance_scale = 25088.0
    normalized_variance = input_var / variance_scale
    epsilon = 1e-05
    adjusted_variance = normalized_variance + epsilon
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(adjusted_variance)
    scaled_data = normalized_data * inv_sqrt_variance
    scaled_and_shifted_data = scaled_data * input_scale
    final_output = scaled_and_shifted_data + input_shift
    
    relu_output = tl.full([1], 0, tl.int32)
    relu_applied_output = triton_helpers.maximum(relu_output, final_output)
    
    tl.store(output_ptr + (element_indices), relu_applied_output, None)