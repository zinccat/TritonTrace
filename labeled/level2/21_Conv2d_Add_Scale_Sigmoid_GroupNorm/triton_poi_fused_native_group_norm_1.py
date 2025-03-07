# From: 21_Conv2d_Add_Scale_Sigmoid_GroupNorm

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_native_group_norm_1(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_bias, 
    input_ptr_rsqrt, input_ptr_shift, input_ptr_add, output_ptr, 
    num_elements, BLOCK_SIZE: tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    index = block_indices
    group_index = (block_indices // 900) % 16
    batch_index = (block_indices // 900)
    
    mean = tl.load(input_ptr_mean + (index), None)
    variance = tl.load(input_ptr_var + (group_index), None, eviction_policy='evict_last')
    scale = tl.load(input_ptr_scale + (group_index), None, eviction_policy='evict_last')
    bias = tl.load(input_ptr_bias + (batch_index // 2), None, eviction_policy='evict_last')
    rsqrt = tl.load(input_ptr_rsqrt + (batch_index // 2), None, eviction_policy='evict_last')
    shift = tl.load(input_ptr_shift + (group_index), None, eviction_policy='evict_last')
    add = tl.load(input_ptr_add + (group_index), None, eviction_policy='evict_last')
    
    normalized = mean + variance
    scaled = normalized * scale
    activated = tl.sigmoid(scaled)
    shifted = activated - bias
    
    variance_scale = 1800.0
    variance_adjusted = rsqrt / variance_scale
    epsilon = 1e-05
    variance_stabilized = variance_adjusted + epsilon
    rsqrt_stabilized = tl.extra.cuda.libdevice.rsqrt(variance_stabilized)
    
    normalized_output = shifted * rsqrt_stabilized
    scaled_output = normalized_output * shift
    final_output = scaled_output + add
    
    tl.store(output_ptr + (index), final_output, None)