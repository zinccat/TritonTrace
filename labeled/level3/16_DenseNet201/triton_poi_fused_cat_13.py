# From: 16_DenseNet201

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_13poi_fused_cat_13(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    channel_index = (x_index // 3136) % 128
    spatial_index = x_index % 3136
    batch_index = x_index // 401408
    linear_index = x_index
    
    channel_condition = channel_index < 64
    load_value_0 = tl.load(in_ptr0 + (spatial_index + 3136 * channel_index + 200704 * batch_index), channel_condition, other=0.0)
    
    channel_condition_1 = channel_index >= 64
    channel_condition_2 = channel_index < 96
    combined_condition = channel_condition_1 & channel_condition_2
    load_value_1 = tl.load(in_ptr1 + (spatial_index + 3136 * (channel_index - 64) + 100352 * batch_index), combined_condition, other=0.0)
    
    channel_condition_3 = channel_index >= 96
    load_value_2 = tl.load(in_ptr2 + (spatial_index + 3136 * (channel_index - 96) + 100352 * batch_index), channel_condition_3, other=0.0)
    
    selected_value_1 = tl.where(combined_condition, load_value_1, load_value_2)
    selected_value = tl.where(channel_condition, load_value_0, selected_value_1)
    
    tl.store(out_ptr0 + (linear_index), selected_value, None)