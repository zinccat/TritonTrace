# From: 92_Conv2d_GroupNorm_Tanh_HardSwish_ResidualAdd_LogSumExp

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_group_norm_tanh_2(
    input_ptr_mean, input_ptr_inv_std, input_ptr_gamma, input_ptr_beta, input_ptr_scale, 
    output_ptr, kernel_size_0, kernel_size_1, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    linear_index = index
    group_index = index // kernel_size_0
    channel_index = ((index // kernel_size_1) % 16)
    
    input_mean = tl.load(input_ptr_mean + (linear_index), mask, eviction_policy='evict_last')
    input_inv_std = tl.load(input_ptr_inv_std + (group_index // 2), mask, eviction_policy='evict_last')
    input_gamma = tl.load(input_ptr_gamma + (group_index // 2), mask, eviction_policy='evict_last')
    input_beta = tl.load(input_ptr_beta + (channel_index), mask, eviction_policy='evict_last')
    input_scale = tl.load(input_ptr_scale + (channel_index), mask, eviction_policy='evict_last')
    
    normalized_input = input_mean - input_inv_std
    scaled_input = normalized_input * input_gamma
    scaled_and_shifted_input = scaled_input * input_beta
    final_input = scaled_and_shifted_input + input_scale
    
    tanh_output = tl.extra.cuda.libdevice.tanh(final_input)
    tl.store(output_ptr + (linear_index), tanh_output, mask)