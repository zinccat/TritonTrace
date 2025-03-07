# From: 16_DenseNet201

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_101poi_fused_cat_101(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, input_ptr5, 
    output_ptr0, num_elements, BLOCK_SIZE : tl.constexpr
):
    num_elements = 815360
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements

    channel_index = (block_indices // 196) % 416
    row_index = block_indices % 196
    depth_index = block_indices // 81536
    linear_index = block_indices

    channel_256 = channel_index
    channel_288 = 288
    channel_320 = 320
    channel_352 = 352
    channel_384 = 384
    channel_416 = 416

    load_mask_256 = channel_256 < channel_256
    load_mask_288 = (channel_256 >= channel_256) & (channel_256 < channel_288)
    load_mask_320 = (channel_256 >= channel_288) & (channel_256 < channel_320)
    load_mask_352 = (channel_256 >= channel_320) & (channel_256 < channel_352)
    load_mask_384 = (channel_256 >= channel_352) & (channel_256 < channel_384)
    load_mask_416 = channel_256 >= channel_384

    data_256 = tl.load(
        input_ptr0 + (row_index + 196 * channel_256 + 50176 * depth_index), 
        load_mask_256 & valid_mask, 
        other=0.0
    )
    data_288 = tl.load(
        input_ptr1 + (row_index + 196 * (channel_256 - 256) + 6272 * depth_index), 
        load_mask_288 & valid_mask, 
        other=0.0
    )
    data_320 = tl.load(
        input_ptr2 + (row_index + 196 * (channel_256 - 288) + 6272 * depth_index), 
        load_mask_320 & valid_mask, 
        other=0.0
    )
    data_352 = tl.load(
        input_ptr3 + (row_index + 196 * (channel_256 - 320) + 6272 * depth_index), 
        load_mask_352 & valid_mask, 
        other=0.0
    )
    data_384 = tl.load(
        input_ptr4 + (row_index + 196 * (channel_256 - 352) + 6272 * depth_index), 
        load_mask_384 & valid_mask, 
        other=0.0
    )
    data_416 = tl.load(
        input_ptr5 + (row_index + 196 * (channel_256 - 384) + 6272 * depth_index), 
        load_mask_416 & valid_mask, 
        other=0.0
    )

    result = tl.where(load_mask_384, data_384, data_416)
    result = tl.where(load_mask_352, data_352, result)
    result = tl.where(load_mask_320, data_320, result)
    result = tl.where(load_mask_288, data_288, result)
    result = tl.where(load_mask_256, data_256, result)

    tl.store(output_ptr0 + (linear_index), result, valid_mask)