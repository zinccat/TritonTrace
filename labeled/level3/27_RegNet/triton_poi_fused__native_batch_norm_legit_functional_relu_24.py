# From: 27_RegNet

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_24poi_fused__native_batch_norm_legit_functional_relu_24(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_shift, input_ptr_input, 
    output_ptr, num_elements, BLOCK_SIZE : tl.constexpr):

    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    index = block_indices
    element_index = index % 256
    
    mean_value = tl.load(input_ptr_mean + (index), None)
    variance_value = tl.load(input_ptr_var + (element_index), None, eviction_policy='evict_last')
    scale_value = tl.load(input_ptr_scale + (element_index), None, eviction_policy='evict_last')
    shift_value = tl.load(input_ptr_shift + (element_index), None, eviction_policy='evict_last')
    input_value = tl.load(input_ptr_input + (element_index), None, eviction_policy='evict_last')
    
    normalized_value = input_value - mean_value
    variance_scale = 25088.0
    variance_adjusted = variance_value / variance_scale
    epsilon = 1e-05
    variance_stabilized = variance_adjusted + epsilon
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_stabilized)
    scaled_normalized_value = normalized_value * inv_sqrt_variance
    scaled_value = scaled_normalized_value * scale_value
    shifted_value = scaled_value + shift_value
    
    relu_output = tl.full([1], 0, tl.int32)
    relu_applied = triton_helpers.maximum(relu_output, shifted_value)
    
    tl.store(output_ptr + (index), relu_applied, None)