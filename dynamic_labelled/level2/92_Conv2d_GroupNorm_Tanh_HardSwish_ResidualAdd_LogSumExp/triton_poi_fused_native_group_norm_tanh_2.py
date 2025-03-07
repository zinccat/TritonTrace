# From: 92_Conv2d_GroupNorm_Tanh_HardSwish_ResidualAdd_LogSumExp

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_group_norm_tanh_2poi_fused_native_group_norm_tanh_2(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, 
    output_ptr0, kernel_size0, kernel_size1, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    element_index = index
    group_index = index // kernel_size0
    channel_index = ((index // kernel_size1) % 16)
    
    input_value0 = tl.load(input_ptr0 + (element_index), mask, eviction_policy='evict_last')
    input_value1 = tl.load(input_ptr1 + (group_index // 2), mask, eviction_policy='evict_last')
    input_value2 = tl.load(input_ptr2 + (group_index // 2), mask, eviction_policy='evict_last')
    input_value3 = tl.load(input_ptr3 + (channel_index), mask, eviction_policy='evict_last')
    input_value4 = tl.load(input_ptr4 + (channel_index), mask, eviction_policy='evict_last')
    
    normalized_value = input_value0 - input_value1
    scaled_value = normalized_value * input_value2
    weighted_value = scaled_value * input_value3
    biased_value = weighted_value + input_value4
    
    tanh_value = tl.extra.cuda.libdevice.tanh(biased_value)
    tl.store(output_ptr0 + (element_index), tanh_value, mask)