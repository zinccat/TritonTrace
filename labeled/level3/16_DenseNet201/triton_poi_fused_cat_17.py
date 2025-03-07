# From: 16_DenseNet201

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_17poi_fused_cat_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    
    channel_index = (x_index // 3136) % 160
    block_index = x_index % 3136
    batch_index = x_index // 501760
    linear_index = x_index
    
    channel = channel_index
    tl.full([1], 0, tl.int64)
    
    channel_64 = tl.full([1], 64, tl.int64)
    load_condition_64 = channel < channel_64
    value_64 = tl.load(in_ptr0 + (block_index + 3136 * channel + 200704 * batch_index), load_condition_64, other=0.0)
    
    channel_96 = tl.full([1], 96, tl.int64)
    load_condition_96 = channel >= channel_64
    load_condition_96 &= channel < channel_96
    value_96 = tl.load(in_ptr1 + (block_index + 3136 * (channel - 64) + 100352 * batch_index), load_condition_96, other=0.0)
    
    channel_128 = tl.full([1], 128, tl.int64)
    load_condition_128 = channel >= channel_96
    load_condition_128 &= channel < channel_128
    value_128 = tl.load(in_ptr2 + (block_index + 3136 * (channel - 96) + 100352 * batch_index), load_condition_128, other=0.0)
    
    channel_160 = tl.full([1], 160, tl.int64)
    load_condition_160 = channel >= channel_128
    value_160 = tl.load(in_ptr3 + (block_index + 3136 * (channel - 128) + 100352 * batch_index), load_condition_160, other=0.0)
    
    selected_value_128_or_160 = tl.where(load_condition_128, value_128, value_160)
    selected_value_96_or_128_or_160 = tl.where(load_condition_96, value_96, selected_value_128_or_160)
    final_value = tl.where(load_condition_64, value_64, selected_value_96_or_128_or_160)
    
    tl.store(out_ptr0 + (linear_index), final_value, None)