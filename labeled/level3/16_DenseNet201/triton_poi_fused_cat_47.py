# From: 16_DenseNet201

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_47poi_fused_cat_47(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, input_ptr5, 
    output_ptr0, num_elements, BLOCK_SIZE : tl.constexpr
):
    num_elements = 2257920
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements

    channel_index = (block_indices // 784) % 288
    pixel_index = block_indices % 784
    batch_index = block_indices // 225792
    linear_index = block_indices

    channel_threshold_1 = 128
    channel_threshold_2 = 160
    channel_threshold_3 = 192
    channel_threshold_4 = 224
    channel_threshold_5 = 256

    load_mask_1 = channel_index < channel_threshold_1
    load_mask_2 = (channel_index >= channel_threshold_1) & (channel_index < channel_threshold_2)
    load_mask_3 = (channel_index >= channel_threshold_2) & (channel_index < channel_threshold_3)
    load_mask_4 = (channel_index >= channel_threshold_3) & (channel_index < channel_threshold_4)
    load_mask_5 = (channel_index >= channel_threshold_4) & (channel_index < channel_threshold_5)
    load_mask_6 = channel_index >= channel_threshold_5

    data_1 = tl.load(input_ptr0 + (pixel_index + 784 * channel_index + 100352 * batch_index), load_mask_1 & valid_mask, other=0.0)
    data_2 = tl.load(input_ptr1 + (pixel_index + 784 * (channel_index - 128) + 25088 * batch_index), load_mask_2 & valid_mask, other=0.0)
    data_3 = tl.load(input_ptr2 + (pixel_index + 784 * (channel_index - 160) + 25088 * batch_index), load_mask_3 & valid_mask, other=0.0)
    data_4 = tl.load(input_ptr3 + (pixel_index + 784 * (channel_index - 192) + 25088 * batch_index), load_mask_4 & valid_mask, other=0.0)
    data_5 = tl.load(input_ptr4 + (pixel_index + 784 * (channel_index - 224) + 25088 * batch_index), load_mask_5 & valid_mask, other=0.0)
    data_6 = tl.load(input_ptr5 + (pixel_index + 784 * (channel_index - 256) + 25088 * batch_index), load_mask_6 & valid_mask, other=0.0)

    combined_data = tl.where(load_mask_5, data_5, data_6)
    combined_data = tl.where(load_mask_4, data_4, combined_data)
    combined_data = tl.where(load_mask_3, data_3, combined_data)
    combined_data = tl.where(load_mask_2, data_2, combined_data)
    combined_data = tl.where(load_mask_1, data_1, combined_data)

    tl.store(output_ptr0 + (linear_index), combined_data, valid_mask)