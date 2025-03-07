# From: 15_DenseNet121

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_95poi_fused_cat_95(
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

    channel_base = x_channel
    channel_mask_256 = channel_base < 256
    value_256 = tl.load(input_ptr0 + (x_row + 196 * x_channel + 50176 * x_depth), channel_mask_256 & xmask, other=0.0)

    channel_mask_288 = (channel_base >= 256) & (channel_base < 288)
    value_288 = tl.load(input_ptr1 + (x_row + 196 * (channel_base - 256) + 6272 * x_depth), channel_mask_288 & xmask, other=0.0)

    channel_mask_320 = (channel_base >= 288) & (channel_base < 320)
    value_320 = tl.load(input_ptr2 + (x_row + 196 * (channel_base - 288) + 6272 * x_depth), channel_mask_320 & xmask, other=0.0)

    channel_mask_352 = (channel_base >= 320) & (channel_base < 352)
    value_352 = tl.load(input_ptr3 + (x_row + 196 * (channel_base - 320) + 6272 * x_depth), channel_mask_352 & xmask, other=0.0)

    channel_mask_384 = (channel_base >= 352) & (channel_base < 384)
    value_384 = tl.load(input_ptr4 + (x_row + 196 * (channel_base - 352) + 6272 * x_depth), channel_mask_384 & xmask, other=0.0)

    channel_mask_416 = (channel_base >= 384) & (channel_base < 416)
    value_416 = tl.load(input_ptr5 + (x_row + 196 * (channel_base - 384) + 6272 * x_depth), channel_mask_416 & xmask, other=0.0)

    channel_mask_448 = (channel_base >= 416) & (channel_base < 448)
    value_448 = tl.load(input_ptr6 + (x_row + 196 * (channel_base - 416) + 6272 * x_depth), channel_mask_448 & xmask, other=0.0)

    channel_mask_480 = channel_base >= 448
    value_480 = tl.load(input_ptr7 + (x_row + 196 * (channel_base - 448) + 6272 * x_depth), channel_mask_480 & xmask, other=0.0)

    result = tl.where(channel_mask_448, value_448, value_480)
    result = tl.where(channel_mask_416, value_416, result)
    result = tl.where(channel_mask_384, value_384, result)
    result = tl.where(channel_mask_352, value_352, result)
    result = tl.where(channel_mask_320, value_320, result)
    result = tl.where(channel_mask_288, value_288, result)
    result = tl.where(channel_mask_256, value_256, result)

    tl.store(output_ptr0 + (x_flat_index), result, xmask)