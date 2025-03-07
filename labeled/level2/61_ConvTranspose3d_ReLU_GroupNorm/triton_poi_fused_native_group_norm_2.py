# From: 61_ConvTranspose3d_ReLU_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused_native_group_norm_2(input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_bias, output_ptr, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    index = block_indices
    group_index = (index // 3240)
    channel_index = (index // 3240) % 128
    
    input_mean = tl.load(input_ptr_mean + (index), None)
    input_var = tl.load(input_ptr_var + ((group_index // 16)), None, eviction_policy='evict_last')
    var_reciprocal = tl.load(input_ptr_scale + ((group_index // 16)), None, eviction_policy='evict_last')
    scale = tl.load(input_ptr_scale + (channel_index), None, eviction_policy='evict_last')
    bias = tl.load(input_ptr_bias + (channel_index), None, eviction_policy='evict_last')
    
    zero = tl.full([1], 0, tl.int32)
    centered_input = triton_helpers.maximum(zero, input_mean) - input_var
    normalization_factor = 51840.0
    epsilon = 1e-05
    
    normalized_input = centered_input * (var_reciprocal / normalization_factor + epsilon).rsqrt()
    scaled_input = normalized_input * scale
    output = scaled_input + bias
    
    tl.store(output_ptr + (index), output, None)