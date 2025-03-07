# From: 16_DenseNet201

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_104poi_fused_cat_104(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, input_ptr5, input_ptr6, 
    output_ptr0, xnumel, XBLOCK: tl.constexpr
):
    xnumel = 878080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel

    # Calculate indices
    channel_index = (xindex // 196) % 448
    spatial_index = xindex % 196
    batch_index = xindex // 87808
    linear_index = xindex

    # Temporary variables for conditions
    zero_tensor = tl.full([1], 0, tl.int64)
    threshold_256 = tl.full([1], 256, tl.int64)
    threshold_288 = tl.full([1], 288, tl.int64)
    threshold_320 = tl.full([1], 320, tl.int64)
    threshold_352 = tl.full([1], 352, tl.int64)
    threshold_384 = tl.full([1], 384, tl.int64)
    threshold_416 = tl.full([1], 416, tl.int64)
    threshold_448 = tl.full([1], 448, tl.int64)

    # Load and conditionally select values
    load_mask_256 = channel_index < threshold_256
    value_256 = tl.load(input_ptr0 + (spatial_index + 196 * channel_index + 50176 * batch_index), load_mask_256 & xmask, other=0.0)

    load_mask_288 = (channel_index >= threshold_256) & (channel_index < threshold_288)
    value_288 = tl.load(input_ptr1 + (spatial_index + 196 * (channel_index - 256) + 6272 * batch_index), load_mask_288 & xmask, other=0.0)

    load_mask_320 = (channel_index >= threshold_288) & (channel_index < threshold_320)
    value_320 = tl.load(input_ptr2 + (spatial_index + 196 * (channel_index - 288) + 6272 * batch_index), load_mask_320 & xmask, other=0.0)

    load_mask_352 = (channel_index >= threshold_320) & (channel_index < threshold_352)
    value_352 = tl.load(input_ptr3 + (spatial_index + 196 * (channel_index - 320) + 6272 * batch_index), load_mask_352 & xmask, other=0.0)

    load_mask_384 = (channel_index >= threshold_352) & (channel_index < threshold_384)
    value_384 = tl.load(input_ptr4 + (spatial_index + 196 * (channel_index - 352) + 6272 * batch_index), load_mask_384 & xmask, other=0.0)

    load_mask_416 = (channel_index >= threshold_384) & (channel_index < threshold_416)
    value_416 = tl.load(input_ptr5 + (spatial_index + 196 * (channel_index - 384) + 6272 * batch_index), load_mask_416 & xmask, other=0.0)

    load_mask_448 = channel_index >= threshold_416
    value_448 = tl.load(input_ptr6 + (spatial_index + 196 * (channel_index - 416) + 6272 * batch_index), load_mask_448 & xmask, other=0.0)

    # Combine values based on conditions
    combined_value = tl.where(load_mask_416, value_416, value_448)
    combined_value = tl.where(load_mask_384, value_384, combined_value)
    combined_value = tl.where(load_mask_352, value_352, combined_value)
    combined_value = tl.where(load_mask_320, value_320, combined_value)
    combined_value = tl.where(load_mask_288, value_288, combined_value)
    combined_value = tl.where(load_mask_256, value_256, combined_value)

    # Store the result
    tl.store(output_ptr0 + (linear_index), combined_value, xmask)