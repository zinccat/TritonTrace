# From: 15_DenseNet121

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_83poi_fused_cat_83(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 689920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel

    # Calculate indices
    channel_index = (xindex // 196) % 352
    height_index = xindex % 196
    batch_index = xindex // 68992
    linear_index = xindex

    # Temporary variables for conditions
    channel_256_mask = channel_index < 256
    channel_288_mask = (channel_index >= 256) & (channel_index < 288)
    channel_320_mask = (channel_index >= 288) & (channel_index < 320)
    channel_352_mask = channel_index >= 320

    # Load values based on conditions
    value_0 = tl.load(in_ptr0 + (height_index + 196 * channel_index + 50176 * batch_index), channel_256_mask & xmask, other=0.0)
    value_1 = tl.load(in_ptr1 + (height_index + 196 * (channel_index - 256) + 6272 * batch_index), channel_288_mask & xmask, other=0.0)
    value_2 = tl.load(in_ptr2 + (height_index + 196 * (channel_index - 288) + 6272 * batch_index), channel_320_mask & xmask, other=0.0)
    value_3 = tl.load(in_ptr3 + (height_index + 196 * (channel_index - 320) + 6272 * batch_index), channel_352_mask & xmask, other=0.0)

    # Combine values based on conditions
    combined_value_2_3 = tl.where(channel_320_mask, value_2, value_3)
    combined_value_1_2_3 = tl.where(channel_288_mask, value_1, combined_value_2_3)
    final_value = tl.where(channel_256_mask, value_0, combined_value_1_2_3)

    # Store the result
    tl.store(out_ptr0 + (linear_index), final_value, xmask)