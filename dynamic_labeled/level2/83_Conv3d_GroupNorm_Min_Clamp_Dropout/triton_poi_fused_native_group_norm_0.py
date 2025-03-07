# From: 83_Conv3d_GroupNorm_Min_Clamp_Dropout

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_group_norm_0(input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, output_ptr0, kernel_size0, kernel_size1, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    linear_index = index
    group_index = index // kernel_size0
    channel_index = ((index // kernel_size1) % 16)
    
    input_data = tl.load(input_ptr0 + (linear_index), mask, eviction_policy='evict_last')
    mean_data = tl.load(input_ptr1 + (group_index // 2), mask, eviction_policy='evict_last')
    variance_data = tl.load(input_ptr2 + (group_index // 2), mask, eviction_policy='evict_last')
    gamma_data = tl.load(input_ptr3 + (channel_index), mask, eviction_policy='evict_last')
    beta_data = tl.load(input_ptr4 + (channel_index), mask, eviction_policy='evict_last')
    
    normalized_data = input_data - mean_data
    scaled_data = normalized_data * variance_data
    scaled_gamma = scaled_data * gamma_data
    output_data = scaled_gamma + beta_data
    
    tl.store(output_ptr0 + (linear_index), output_data, mask)