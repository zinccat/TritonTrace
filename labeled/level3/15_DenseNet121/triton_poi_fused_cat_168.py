# From: 15_DenseNet121

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_168poi_fused_cat_168(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, input_ptr5, input_ptr6, input_ptr7,
    output_ptr0, xnumel, XBLOCK: tl.constexpr
):
    xnumel = 360640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x_channel = (xindex // 49) % 736
    x_row = xindex % 49
    x_depth = xindex // 36064
    x_flat_index = xindex

    channel_base = x_channel
    channel_mask_512 = channel_base < 512
    value_512 = tl.load(input_ptr0 + (x_row + 49 * x_channel + 25088 * x_depth), channel_mask_512 & xmask, other=0.0)

    channel_mask_544 = (channel_base >= 512) & (channel_base < 544)
    value_544 = tl.load(input_ptr1 + (x_row + 49 * (channel_base - 512) + 1568 * x_depth), channel_mask_544 & xmask, other=0.0)

    channel_mask_576 = (channel_base >= 544) & (channel_base < 576)
    value_576 = tl.load(input_ptr2 + (x_row + 49 * (channel_base - 544) + 1568 * x_depth), channel_mask_576 & xmask, other=0.0)

    channel_mask_608 = (channel_base >= 576) & (channel_base < 608)
    value_608 = tl.load(input_ptr3 + (x_row + 49 * (channel_base - 576) + 1568 * x_depth), channel_mask_608 & xmask, other=0.0)

    channel_mask_640 = (channel_base >= 608) & (channel_base < 640)
    value_640 = tl.load(input_ptr4 + (x_row + 49 * (channel_base - 608) + 1568 * x_depth), channel_mask_640 & xmask, other=0.0)

    channel_mask_672 = (channel_base >= 640) & (channel_base < 672)
    value_672 = tl.load(input_ptr5 + (x_row + 49 * (channel_base - 640) + 1568 * x_depth), channel_mask_672 & xmask, other=0.0)

    channel_mask_704 = (channel_base >= 672) & (channel_base < 704)
    value_704 = tl.load(input_ptr6 + (x_row + 49 * (channel_base - 672) + 1568 * x_depth), channel_mask_704 & xmask, other=0.0)

    channel_mask_736 = channel_base >= 704
    value_736 = tl.load(input_ptr7 + (x_row + 49 * (channel_base - 704) + 1568 * x_depth), channel_mask_736 & xmask, other=0.0)

    result = tl.where(channel_mask_704, value_704, value_736)
    result = tl.where(channel_mask_672, value_672, result)
    result = tl.where(channel_mask_640, value_640, result)
    result = tl.where(channel_mask_608, value_608, result)
    result = tl.where(channel_mask_576, value_576, result)
    result = tl.where(channel_mask_544, value_544, result)
    result = tl.where(channel_mask_512, value_512, result)

    tl.store(output_ptr0 + (x_flat_index), result, xmask)