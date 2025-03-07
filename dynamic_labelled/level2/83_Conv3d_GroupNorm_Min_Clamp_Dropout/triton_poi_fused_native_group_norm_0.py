# From: 83_Conv3d_GroupNorm_Min_Clamp_Dropout

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_group_norm_0poi_fused_native_group_norm_0(
    input_ptr_mean, input_ptr_inv_std, input_ptr_var, input_ptr_gamma, input_ptr_beta, 
    output_ptr, kernel_size_0, kernel_size_1, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    linear_index = index
    group_index = index // kernel_size_0
    channel_index = ((index // kernel_size_1) % 16)
    
    mean = tl.load(input_ptr_mean + (linear_index), mask, eviction_policy='evict_last')
    inv_std = tl.load(input_ptr_inv_std + (group_index // 2), mask, eviction_policy='evict_last')
    var = tl.load(input_ptr_var + (group_index // 2), mask, eviction_policy='evict_last')
    gamma = tl.load(input_ptr_gamma + (channel_index), mask, eviction_policy='evict_last')
    beta = tl.load(input_ptr_beta + (channel_index), mask, eviction_policy='evict_last')
    
    normalized = mean - inv_std
    scaled = normalized * var
    scaled_gamma = scaled * gamma
    output = scaled_gamma + beta
    
    tl.store(output_ptr + (linear_index), output, mask)