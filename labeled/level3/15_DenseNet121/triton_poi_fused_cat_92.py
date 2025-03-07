# From: 15_DenseNet121

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_92poi_fused_cat_92(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, input_ptr5, input_ptr6, 
    output_ptr0, num_elements, BLOCK_SIZE : tl.constexpr
):
    num_elements = 878080
    offset = tl.program_id(0) * BLOCK_SIZE
    index = offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = index < num_elements

    channel_index = (index // 196) % 448
    row_index = index % 196
    batch_index = index // 87808
    flat_index = index

    zero_value = tl.full([1], 0, tl.int64)
    channel_256 = tl.full([1], 256, tl.int64)
    channel_288 = tl.full([1], 288, tl.int64)
    channel_320 = tl.full([1], 320, tl.int64)
    channel_352 = tl.full([1], 352, tl.int64)
    channel_384 = tl.full([1], 384, tl.int64)
    channel_416 = tl.full([1], 416, tl.int64)
    channel_448 = tl.full([1], 448, tl.int64)

    load_mask_256 = channel_index < channel_256
    value_256 = tl.load(input_ptr0 + (row_index + 196 * channel_index + 50176 * batch_index), load_mask_256 & mask, other=0.0)

    load_mask_288 = (channel_index >= channel_256) & (channel_index < channel_288)
    value_288 = tl.load(input_ptr1 + (row_index + 196 * ((-256) + channel_index) + 6272 * batch_index), load_mask_288 & mask, other=0.0)

    load_mask_320 = (channel_index >= channel_288) & (channel_index < channel_320)
    value_320 = tl.load(input_ptr2 + (row_index + 196 * ((-288) + channel_index) + 6272 * batch_index), load_mask_320 & mask, other=0.0)

    load_mask_352 = (channel_index >= channel_320) & (channel_index < channel_352)
    value_352 = tl.load(input_ptr3 + (row_index + 196 * ((-320) + channel_index) + 6272 * batch_index), load_mask_352 & mask, other=0.0)

    load_mask_384 = (channel_index >= channel_352) & (channel_index < channel_384)
    value_384 = tl.load(input_ptr4 + (row_index + 196 * ((-352) + channel_index) + 6272 * batch_index), load_mask_384 & mask, other=0.0)

    load_mask_416 = (channel_index >= channel_384) & (channel_index < channel_416)
    value_416 = tl.load(input_ptr5 + (row_index + 196 * ((-384) + channel_index) + 6272 * batch_index), load_mask_416 & mask, other=0.0)

    load_mask_448 = channel_index >= channel_416
    value_448 = tl.load(input_ptr6 + (row_index + 196 * ((-416) + channel_index) + 6272 * batch_index), load_mask_448 & mask, other=0.0)

    result = tl.where(load_mask_416, value_416, value_448)
    result = tl.where(load_mask_384, value_384, result)
    result = tl.where(load_mask_352, value_352, result)
    result = tl.where(load_mask_320, value_320, result)
    result = tl.where(load_mask_288, value_288, result)
    result = tl.where(load_mask_256, value_256, result)

    tl.store(output_ptr0 + (flat_index), result, mask)