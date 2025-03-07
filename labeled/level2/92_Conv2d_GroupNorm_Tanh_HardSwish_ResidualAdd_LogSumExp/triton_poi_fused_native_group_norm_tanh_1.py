# From: 92_Conv2d_GroupNorm_Tanh_HardSwish_ResidualAdd_LogSumExp

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused_native_group_norm_tanh_1(
    input_ptr_mean, input_ptr_var, input_ptr_gamma, input_ptr_beta, input_ptr_scale, 
    output_ptr, num_elements, BLOCK_SIZE : tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    element_index = block_indices
    group_index = element_index // 900
    channel_index = group_index % 16
    
    input_mean = tl.load(input_ptr_mean + (element_index), None)
    input_var = tl.load(input_ptr_var + (group_index // 2), None, eviction_policy='evict_last')
    input_gamma = tl.load(input_ptr_gamma + (group_index // 2), None, eviction_policy='evict_last')
    input_beta = tl.load(input_ptr_beta + (channel_index), None, eviction_policy='evict_last')
    input_scale = tl.load(input_ptr_scale + (channel_index), None, eviction_policy='evict_last')
    
    normalized_input = input_mean - input_var
    scaled_input = normalized_input * input_gamma
    gamma_scaled_input = scaled_input * input_beta
    biased_input = gamma_scaled_input + input_scale
    
    tanh_output = tl.extra.cuda.libdevice.tanh(biased_input)
    tl.store(output_ptr + (element_index), tanh_output, None)