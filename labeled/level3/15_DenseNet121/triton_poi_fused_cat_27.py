# From: 15_DenseNet121

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_27poi_fused_cat_27(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, input_ptr5, input_ptr6, 
    output_ptr0, num_elements, BLOCK_SIZE : tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    
    channel_index = (block_indices // 3136) % 256
    spatial_index = block_indices % 3136
    batch_index = block_indices // 802816
    linear_index = block_indices
    
    current_channel = channel_index
    
    tl.full([1], 0, tl.int64)
    channel_threshold_64 = tl.full([1], 64, tl.int64)
    is_channel_less_than_64 = current_channel < channel_threshold_64
    value_0 = tl.load(input_ptr0 + (spatial_index + 3136 * channel_index + 200704 * batch_index), is_channel_less_than_64, other=0.0)
    
    is_channel_greater_equal_64 = current_channel >= channel_threshold_64
    channel_threshold_96 = tl.full([1], 96, tl.int64)
    is_channel_less_than_96 = current_channel < channel_threshold_96
    is_channel_between_64_and_96 = is_channel_greater_equal_64 & is_channel_less_than_96
    value_1 = tl.load(input_ptr1 + (spatial_index + 3136 * ((-64) + channel_index) + 100352 * batch_index), is_channel_between_64_and_96, other=0.0)
    
    is_channel_greater_equal_96 = current_channel >= channel_threshold_96
    channel_threshold_128 = tl.full([1], 128, tl.int64)
    is_channel_less_than_128 = current_channel < channel_threshold_128
    is_channel_between_96_and_128 = is_channel_greater_equal_96 & is_channel_less_than_128
    value_2 = tl.load(input_ptr2 + (spatial_index + 3136 * ((-96) + channel_index) + 100352 * batch_index), is_channel_between_96_and_128, other=0.0)
    
    is_channel_greater_equal_128 = current_channel >= channel_threshold_128
    channel_threshold_160 = tl.full([1], 160, tl.int64)
    is_channel_less_than_160 = current_channel < channel_threshold_160
    is_channel_between_128_and_160 = is_channel_greater_equal_128 & is_channel_less_than_160
    value_3 = tl.load(input_ptr3 + (spatial_index + 3136 * ((-128) + channel_index) + 100352 * batch_index), is_channel_between_128_and_160, other=0.0)
    
    is_channel_greater_equal_160 = current_channel >= channel_threshold_160
    channel_threshold_192 = tl.full([1], 192, tl.int64)
    is_channel_less_than_192 = current_channel < channel_threshold_192
    is_channel_between_160_and_192 = is_channel_greater_equal_160 & is_channel_less_than_192
    value_4 = tl.load(input_ptr4 + (spatial_index + 3136 * ((-160) + channel_index) + 100352 * batch_index), is_channel_between_160_and_192, other=0.0)
    
    is_channel_greater_equal_192 = current_channel >= channel_threshold_192
    channel_threshold_224 = tl.full([1], 224, tl.int64)
    is_channel_less_than_224 = current_channel < channel_threshold_224
    is_channel_between_192_and_224 = is_channel_greater_equal_192 & is_channel_less_than_224
    value_5 = tl.load(input_ptr5 + (spatial_index + 3136 * ((-192) + channel_index) + 100352 * batch_index), is_channel_between_192_and_224, other=0.0)
    
    is_channel_greater_equal_224 = current_channel >= channel_threshold_224
    channel_threshold_256 = tl.full([1], 256, tl.int64)
    value_6 = tl.load(input_ptr6 + (spatial_index + 3136 * ((-224) + channel_index) + 100352 * batch_index), is_channel_greater_equal_224, other=0.0)
    
    selected_value_5_or_6 = tl.where(is_channel_between_192_and_224, value_5, value_6)
    selected_value_4_or_5_6 = tl.where(is_channel_between_160_and_192, value_4, selected_value_5_or_6)
    selected_value_3_or_4_5_6 = tl.where(is_channel_between_128_and_160, value_3, selected_value_4_or_5_6)
    selected_value_2_or_3_4_5_6 = tl.where(is_channel_between_96_and_128, value_2, selected_value_3_or_4_5_6)
    selected_value_1_or_2_3_4_5_6 = tl.where(is_channel_between_64_and_96, value_1, selected_value_2_or_3_4_5_6)
    final_selected_value = tl.where(is_channel_less_than_64, value_0, selected_value_1_or_2_3_4_5_6)
    
    tl.store(output_ptr0 + (linear_index), final_selected_value, None)