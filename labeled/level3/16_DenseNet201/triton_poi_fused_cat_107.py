# From: 16_DenseNet201

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_107poi_fused_cat_107(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, input_ptr5, input_ptr6, input_ptr7,
    output_ptr0, xnumel, XBLOCK: tl.constexpr
):
    xnumel = 940800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel

    x_channel = (xindex // 196) % 480
    x_row = xindex % 196
    x_depth = xindex // 94080
    x_flat_index = xindex

    channel_value = x_channel
    zero_value = tl.full([1], 0, tl.int64)
    channel_256 = tl.full([1], 256, tl.int64)
    channel_288 = tl.full([1], 288, tl.int64)
    channel_320 = tl.full([1], 320, tl.int64)
    channel_352 = tl.full([1], 352, tl.int64)
    channel_384 = tl.full([1], 384, tl.int64)
    channel_416 = tl.full([1], 416, tl.int64)
    channel_448 = tl.full([1], 448, tl.int64)
    channel_480 = tl.full([1], 480, tl.int64)

    load_mask_256 = channel_value < channel_256
    value_256 = tl.load(input_ptr0 + (x_row + 196 * x_channel + 50176 * x_depth), load_mask_256 & xmask, other=0.0)

    load_mask_288 = (channel_value >= channel_256) & (channel_value < channel_288)
    value_288 = tl.load(input_ptr1 + (x_row + 196 * ((-256) + x_channel) + 6272 * x_depth), load_mask_288 & xmask, other=0.0)

    load_mask_320 = (channel_value >= channel_288) & (channel_value < channel_320)
    value_320 = tl.load(input_ptr2 + (x_row + 196 * ((-288) + x_channel) + 6272 * x_depth), load_mask_320 & xmask, other=0.0)

    load_mask_352 = (channel_value >= channel_320) & (channel_value < channel_352)
    value_352 = tl.load(input_ptr3 + (x_row + 196 * ((-320) + x_channel) + 6272 * x_depth), load_mask_352 & xmask, other=0.0)

    load_mask_384 = (channel_value >= channel_352) & (channel_value < channel_384)
    value_384 = tl.load(input_ptr4 + (x_row + 196 * ((-352) + x_channel) + 6272 * x_depth), load_mask_384 & xmask, other=0.0)

    load_mask_416 = (channel_value >= channel_384) & (channel_value < channel_416)
    value_416 = tl.load(input_ptr5 + (x_row + 196 * ((-384) + x_channel) + 6272 * x_depth), load_mask_416 & xmask, other=0.0)

    load_mask_448 = (channel_value >= channel_416) & (channel_value < channel_448)
    value_448 = tl.load(input_ptr6 + (x_row + 196 * ((-416) + x_channel) + 6272 * x_depth), load_mask_448 & xmask, other=0.0)

    load_mask_480 = channel_value >= channel_448
    value_480 = tl.load(input_ptr7 + (x_row + 196 * ((-448) + x_channel) + 6272 * x_depth), load_mask_480 & xmask, other=0.0)

    result = tl.where(load_mask_448, value_448, value_480)
    result = tl.where(load_mask_416, value_416, result)
    result = tl.where(load_mask_384, value_384, result)
    result = tl.where(load_mask_352, value_352, result)
    result = tl.where(load_mask_320, value_320, result)
    result = tl.where(load_mask_288, value_288, result)
    result = tl.where(load_mask_256, value_256, result)

    tl.store(output_ptr0 + (x_flat_index), result, xmask)