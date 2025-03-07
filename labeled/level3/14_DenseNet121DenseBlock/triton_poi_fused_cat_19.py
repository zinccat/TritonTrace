# From: 14_DenseNet121DenseBlock

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_19poi_fused_cat_19(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, input_ptr5, 
    output_ptr0, num_elements, BLOCK_SIZE : tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    channel_index = (block_indices // 50176) % 192
    base_index = block_indices % 50176
    batch_index = block_indices // 9633792
    global_index = block_indices
    
    channel = channel_index
    tl.full([1], 0, tl.int64)
    
    channel_32 = tl.full([1], 32, tl.int64)
    load_condition_32 = channel < channel_32
    value_32 = tl.load(input_ptr0 + (base_index + 50176 * channel + 1605632 * batch_index), load_condition_32, other=0.0)
    
    channel_64 = tl.full([1], 64, tl.int64)
    load_condition_64 = (channel >= channel_32) & (channel < channel_64)
    value_64 = tl.load(input_ptr1 + (base_index + 50176 * (channel - 32) + 1605632 * batch_index), load_condition_64, other=0.0)
    
    channel_96 = tl.full([1], 96, tl.int64)
    load_condition_96 = (channel >= channel_64) & (channel < channel_96)
    value_96 = tl.load(input_ptr2 + (base_index + 50176 * (channel - 64) + 1605632 * batch_index), load_condition_96, other=0.0)
    
    channel_128 = tl.full([1], 128, tl.int64)
    load_condition_128 = (channel >= channel_96) & (channel < channel_128)
    value_128 = tl.load(input_ptr3 + (base_index + 50176 * (channel - 96) + 1605632 * batch_index), load_condition_128, other=0.0)
    
    channel_160 = tl.full([1], 160, tl.int64)
    load_condition_160 = (channel >= channel_128) & (channel < channel_160)
    value_160 = tl.load(input_ptr4 + (base_index + 50176 * (channel - 128) + 1605632 * batch_index), load_condition_160, other=0.0)
    
    channel_192 = tl.full([1], 192, tl.int64)
    load_condition_192 = channel >= channel_160
    value_192 = tl.load(input_ptr5 + (base_index + 50176 * (channel - 160) + 1605632 * batch_index), load_condition_192, other=0.0)
    
    result_160 = tl.where(load_condition_160, value_160, value_192)
    result_128 = tl.where(load_condition_128, value_128, result_160)
    result_96 = tl.where(load_condition_96, value_96, result_128)
    result_64 = tl.where(load_condition_64, value_64, result_96)
    result_32 = tl.where(load_condition_32, value_32, result_64)
    
    tl.store(output_ptr0 + (global_index), result_32, None)