# From: 15_DenseNet121

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_21poi_fused_cat_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    # Calculate indices
    channel_index = (xindex // 3136) % 192
    spatial_index = xindex % 3136
    batch_index = xindex // 602112
    linear_index = xindex
    
    # Temporary variables for conditions
    channel_limit_1 = 64
    channel_limit_2 = 96
    channel_limit_3 = 128
    channel_limit_4 = 160
    channel_limit_5 = 192
    
    # Load values based on conditions
    load_condition_1 = channel_index < channel_limit_1
    value_1 = tl.load(in_ptr0 + (spatial_index + 3136 * channel_index + 200704 * batch_index), load_condition_1, other=0.0)
    
    load_condition_2 = (channel_index >= channel_limit_1) & (channel_index < channel_limit_2)
    value_2 = tl.load(in_ptr1 + (spatial_index + 3136 * (channel_index - channel_limit_1) + 100352 * batch_index), load_condition_2, other=0.0)
    
    load_condition_3 = (channel_index >= channel_limit_2) & (channel_index < channel_limit_3)
    value_3 = tl.load(in_ptr2 + (spatial_index + 3136 * (channel_index - channel_limit_2) + 100352 * batch_index), load_condition_3, other=0.0)
    
    load_condition_4 = (channel_index >= channel_limit_3) & (channel_index < channel_limit_4)
    value_4 = tl.load(in_ptr3 + (spatial_index + 3136 * (channel_index - channel_limit_3) + 100352 * batch_index), load_condition_4, other=0.0)
    
    load_condition_5 = (channel_index >= channel_limit_4) & (channel_index < channel_limit_5)
    value_5 = tl.load(in_ptr4 + (spatial_index + 3136 * (channel_index - channel_limit_4) + 100352 * batch_index), load_condition_5, other=0.0)
    
    # Select values based on conditions
    selected_value = tl.where(load_condition_5, value_5, tl.where(load_condition_4, value_4, tl.where(load_condition_3, value_3, tl.where(load_condition_2, value_2, value_1))))
    
    # Store the result
    tl.store(out_ptr0 + (linear_index), selected_value, None)