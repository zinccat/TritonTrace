# From: 15_DenseNet121

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_32poi_fused__native_batch_norm_legit_functional_relu_32(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_shift, input_ptr_input, 
    output_ptr, num_elements, BLOCK_SIZE: tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    input_indices = block_indices
    batch_indices = (block_indices // 784) % 128
    
    input_data = tl.load(input_ptr_input + (input_indices), None)
    mean_data = tl.load(input_ptr_mean + (batch_indices), None, eviction_policy='evict_last')
    var_data = tl.load(input_ptr_var + (batch_indices), None, eviction_policy='evict_last')
    scale_data = tl.load(input_ptr_scale + (batch_indices), None, eviction_policy='evict_last')
    shift_data = tl.load(input_ptr_shift + (batch_indices), None, eviction_policy='evict_last')
    
    normalized_data = input_data - mean_data
    variance_scale = 7840.0
    epsilon = 1e-05
    adjusted_variance = var_data / variance_scale
    variance_with_epsilon = adjusted_variance + epsilon
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_with_epsilon)
    
    scaled_data = normalized_data * inv_sqrt_variance
    scaled_and_shifted_data = scaled_data * scale_data
    final_output = scaled_and_shifted_data + shift_data
    
    zero_tensor = tl.full([1], 0, tl.int32)
    relu_output = triton_helpers.maximum(zero_tensor, final_output)
    
    tl.store(output_ptr + (input_indices), relu_output, None)