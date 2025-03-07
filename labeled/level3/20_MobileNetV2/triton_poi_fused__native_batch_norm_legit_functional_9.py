# From: 20_MobileNetV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_9poi_fused__native_batch_norm_legit_functional_9(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_bias, input_ptr_x, 
    output_ptr, num_elements, BLOCK_SIZE : tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    element_indices = block_indices
    element_index_mod = (block_indices % 16)
    
    mean_value = tl.load(input_ptr_mean + (element_indices), None)
    variance_value = tl.load(input_ptr_var + (element_index_mod), None, eviction_policy='evict_last')
    scale_value = tl.load(input_ptr_scale + (element_index_mod), None, eviction_policy='evict_last')
    bias_value = tl.load(input_ptr_bias + (element_index_mod), None, eviction_policy='evict_last')
    x_value = tl.load(input_ptr_x + (element_indices), None)
    
    x_centered = x_value - mean_value
    variance_normalized = 125440.0
    variance_adjusted = variance_value / variance_normalized
    epsilon = 1e-05
    variance_stabilized = variance_adjusted + epsilon
    variance_reciprocal_sqrt = tl.extra.cuda.libdevice.rsqrt(variance_stabilized)
    
    x_normalized = x_centered * variance_reciprocal_sqrt
    x_scaled = x_normalized * scale_value
    output_value = x_scaled + bias_value
    
    tl.store(output_ptr + (element_indices), output_value, None)