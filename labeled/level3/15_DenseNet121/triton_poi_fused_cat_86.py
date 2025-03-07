# From: 15_DenseNet121

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_86poi_fused_cat_86(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel

    # Calculate indices for accessing input pointers
    channel_index = (xindex // 196) % 384
    spatial_index = xindex % 196
    batch_index = xindex // 75264
    linear_index = xindex

    # Temporary variables for channel thresholds
    channel_256 = 256
    channel_288 = 288
    channel_320 = 320
    channel_352 = 352
    channel_384 = 384

    # Load data from input pointers based on channel thresholds
    load_mask_256 = channel_index < channel_256
    data_256 = tl.load(in_ptr0 + (spatial_index + 196 * channel_index + 50176 * batch_index), load_mask_256 & xmask, other=0.0)

    load_mask_288 = (channel_index >= channel_256) & (channel_index < channel_288)
    data_288 = tl.load(in_ptr1 + (spatial_index + 196 * (channel_index - channel_256) + 6272 * batch_index), load_mask_288 & xmask, other=0.0)

    load_mask_320 = (channel_index >= channel_288) & (channel_index < channel_320)
    data_320 = tl.load(in_ptr2 + (spatial_index + 196 * (channel_index - channel_288) + 6272 * batch_index), load_mask_320 & xmask, other=0.0)

    load_mask_352 = (channel_index >= channel_320) & (channel_index < channel_352)
    data_352 = tl.load(in_ptr3 + (spatial_index + 196 * (channel_index - channel_320) + 6272 * batch_index), load_mask_352 & xmask, other=0.0)

    load_mask_384 = channel_index >= channel_352
    data_384 = tl.load(in_ptr4 + (spatial_index + 196 * (channel_index - channel_352) + 6272 * batch_index), load_mask_384 & xmask, other=0.0)

    # Combine data based on channel thresholds
    combined_data = tl.where(load_mask_352, data_352, data_384)
    combined_data = tl.where(load_mask_320, data_320, combined_data)
    combined_data = tl.where(load_mask_288, data_288, combined_data)
    combined_data = tl.where(load_mask_256, data_256, combined_data)

    # Store the result in the output pointer
    tl.store(out_ptr0 + (linear_index), combined_data, xmask)