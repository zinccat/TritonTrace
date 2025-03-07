# From: 16_DenseNet201

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_35poi_fused_cat_35(input_ptr0, input_ptr1, output_ptr0, num_elements, BLOCK_SIZE : tl.constexpr):
    num_elements = 1254400
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    
    channel_index = (block_indices // 784) % 160
    pixel_index = block_indices % 784
    batch_index = block_indices // 125440
    linear_index = block_indices
    
    channel_threshold = 128
    max_channel_index = 160
    
    is_below_threshold = channel_index < channel_threshold
    load_mask_below = is_below_threshold & valid_mask
    
    data_below_threshold = tl.load(
        input_ptr0 + (pixel_index + 784 * channel_index + 100352 * batch_index),
        load_mask_below,
        other=0.0
    )
    
    is_above_threshold = channel_index >= channel_threshold
    load_mask_above = is_above_threshold & valid_mask
    
    data_above_threshold = tl.load(
        input_ptr1 + (pixel_index + 784 * ((-128) + channel_index) + 25088 * batch_index),
        load_mask_above,
        other=0.0
    )
    
    selected_data = tl.where(is_below_threshold, data_below_threshold, data_above_threshold)
    
    tl.store(output_ptr0 + (linear_index), selected_data, valid_mask)