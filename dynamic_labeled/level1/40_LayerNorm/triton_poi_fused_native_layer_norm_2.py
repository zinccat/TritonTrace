# From: 40_LayerNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_layer_norm_2(input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_bias, input_ptr_input, output_ptr, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    block_index = block_indices
    block_index_large = block_indices // 4194304
    block_index_small = block_indices % 4194304
    
    mean_value = tl.load(input_ptr_mean + (block_index), None)
    variance_value = tl.load(input_ptr_var + (block_index_large), None, eviction_policy='evict_last')
    scale_value = tl.load(input_ptr_scale + (block_index_large), None, eviction_policy='evict_last')
    bias_value = tl.load(input_ptr_bias + (block_index_small), None, eviction_policy='evict_last')
    input_value = tl.load(input_ptr_input + (block_index_small), None, eviction_policy='evict_last')
    
    normalized_value = input_value - mean_value
    scaled_variance = normalized_value * variance_value
    scaled_value = scaled_variance * scale_value
    output_value = scaled_value + bias_value
    
    tl.store(output_ptr + (block_index), output_value, None)