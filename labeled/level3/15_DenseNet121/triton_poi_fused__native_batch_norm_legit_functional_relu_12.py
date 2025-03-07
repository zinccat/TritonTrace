# From: 15_DenseNet121

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_12poi_fused__native_batch_norm_legit_functional_relu_12(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_shift, output_ptr, num_elements, BLOCK_SIZE: tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    index = block_indices
    channel_index = (index // 3136) % 96
    
    input_mean = tl.load(input_ptr_mean + (index), None)
    input_var = tl.load(input_ptr_var + (channel_index), None, eviction_policy='evict_last')
    scale = tl.load(input_ptr_scale + (channel_index), None, eviction_policy='evict_last')
    shift = tl.load(input_ptr_shift + (channel_index), None, eviction_policy='evict_last')
    
    normalized_input = input_mean - input_var
    variance_factor = 31360.0
    epsilon = 1e-05
    
    normalized_variance = input_var / variance_factor
    adjusted_variance = normalized_variance + epsilon
    reciprocal_sqrt = tl.extra.cuda.libdevice.rsqrt(adjusted_variance)
    
    scaled_input = normalized_input * reciprocal_sqrt
    scaled_and_shifted_input = scaled_input * scale
    final_output = scaled_and_shifted_input + shift
    
    zero_tensor = tl.full([1], 0, tl.int32)
    relu_output = triton_helpers.maximum(zero_tensor, final_output)
    
    tl.store(output_ptr + (index), relu_output, None)