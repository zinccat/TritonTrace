# From: 15_DenseNet121

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_159poi_fused_cat_159(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, 
    output_ptr0, xnumel, XBLOCK: tl.constexpr
):
    xnumel = 313600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel

    # Calculate indices
    channel_index = (xindex // 49) % 640
    row_index = xindex % 49
    batch_index = xindex // 31360
    linear_index = xindex

    # Temporary variables for channel boundaries
    channel_512 = channel_index
    channel_544 = channel_index
    channel_576 = channel_index
    channel_608 = channel_index

    # Load values with masking
    load_mask_512 = channel_512 < 512
    value_512 = tl.load(input_ptr0 + (row_index + 49 * channel_512 + 25088 * batch_index), load_mask_512 & xmask, other=0.0)

    load_mask_544 = (channel_512 >= 512) & (channel_544 < 544)
    value_544 = tl.load(input_ptr1 + (row_index + 49 * (channel_544 - 512) + 1568 * batch_index), load_mask_544 & xmask, other=0.0)

    load_mask_576 = (channel_544 >= 544) & (channel_576 < 576)
    value_576 = tl.load(input_ptr2 + (row_index + 49 * (channel_576 - 544) + 1568 * batch_index), load_mask_576 & xmask, other=0.0)

    load_mask_608 = (channel_576 >= 576) & (channel_608 < 608)
    value_608 = tl.load(input_ptr3 + (row_index + 49 * (channel_608 - 576) + 1568 * batch_index), load_mask_608 & xmask, other=0.0)

    load_mask_640 = channel_608 >= 608
    value_640 = tl.load(input_ptr4 + (row_index + 49 * (channel_608 - 608) + 1568 * batch_index), load_mask_640 & xmask, other=0.0)

    # Select values based on channel index
    selected_value = tl.where(load_mask_608, value_608, value_640)
    selected_value = tl.where(load_mask_576, value_576, selected_value)
    selected_value = tl.where(load_mask_544, value_544, selected_value)
    selected_value = tl.where(load_mask_512, value_512, selected_value)

    # Store the result
    tl.store(output_ptr0 + (linear_index), selected_value, xmask)