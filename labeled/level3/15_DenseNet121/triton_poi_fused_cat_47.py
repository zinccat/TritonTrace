# From: 15_DenseNet121

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

    channel_threshold_128 = 128
    channel_threshold_160 = 160
    channel_threshold_192 = 192
    channel_threshold_224 = 224
    channel_threshold_256 = 256

    load_mask_128 = channel_index < channel_threshold_128
    value_128 = tl.load(input_ptr0 + (pixel_index + 784 * channel_index + 100352 * batch_index), load_mask_128 & valid_mask, other=0.0)

    load_mask_160 = (channel_index >= channel_threshold_128) & (channel_index < channel_threshold_160)
    value_160 = tl.load(input_ptr1 + (pixel_index + 784 * (channel_index - 128) + 25088 * batch_index), load_mask_160 & valid_mask, other=0.0)

    load_mask_192 = (channel_index >= channel_threshold_160) & (channel_index < channel_threshold_192)
    value_192 = tl.load(input_ptr2 + (pixel_index + 784 * (channel_index - 160) + 25088 * batch_index), load_mask_192 & valid_mask, other=0.0)

    load_mask_224 = (channel_index >= channel_threshold_192) & (channel_index < channel_threshold_224)
    value_224 = tl.load(input_ptr3 + (pixel_index + 784 * (channel_index - 192) + 25088 * batch_index), load_mask_224 & valid_mask, other=0.0)

    load_mask_256 = (channel_index >= channel_threshold_224) & (channel_index < channel_threshold_256)
    value_256 = tl.load(input_ptr4 + (pixel_index + 784 * (channel_index - 224) + 25088 * batch_index), load_mask_256 & valid_mask, other=0.0)

    load_mask_288 = channel_index >= channel_threshold_256
    value_288 = tl.load(input_ptr5 + (pixel_index + 784 * (channel_index - 256) + 25088 * batch_index), load_mask_288 & valid_mask, other=0.0)

    result = tl.where(load_mask_256, value_256, value_288)
    result = tl.where(load_mask_224, value_224, result)
    result = tl.where(load_mask_192, value_192, result)
    result = tl.where(load_mask_160, value_160, result)
    result = tl.where(load_mask_128, value_128, result)

    tl.store(output_ptr0 + (linear_index), result, valid_mask)