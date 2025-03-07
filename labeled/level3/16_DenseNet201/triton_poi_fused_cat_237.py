# From: 16_DenseNet201

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_237poi_fused_cat_237(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 501760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel

    # Calculate indices
    channel_index = (xindex // 49) % 1024
    spatial_index = xindex % 49
    batch_index = xindex // 50176
    linear_index = xindex

    # Temporary variables for conditions
    channel_base = channel_index
    channel_896 = 896
    channel_928 = 928
    channel_960 = 960
    channel_992 = 992
    channel_1024 = 1024

    # Load and conditionally select values
    load_mask_0 = channel_base < channel_896
    value_0 = tl.load(in_ptr0 + (spatial_index + 49 * channel_base + 43904 * batch_index), load_mask_0 & xmask, other=0.0)

    load_mask_1 = (channel_base >= channel_896) & (channel_base < channel_928)
    value_1 = tl.load(in_ptr1 + (spatial_index + 49 * ((-896) + channel_base) + 1568 * batch_index), load_mask_1 & xmask, other=0.0)

    load_mask_2 = (channel_base >= channel_928) & (channel_base < channel_960)
    value_2 = tl.load(in_ptr2 + (spatial_index + 49 * ((-928) + channel_base) + 1568 * batch_index), load_mask_2 & xmask, other=0.0)

    load_mask_3 = (channel_base >= channel_960) & (channel_base < channel_992)
    value_3 = tl.load(in_ptr3 + (spatial_index + 49 * ((-960) + channel_base) + 1568 * batch_index), load_mask_3 & xmask, other=0.0)

    load_mask_4 = channel_base >= channel_992
    value_4 = tl.load(in_ptr4 + (spatial_index + 49 * ((-992) + channel_base) + 1568 * batch_index), load_mask_4 & xmask, other=0.0)

    # Conditional selection
    selected_value_3_4 = tl.where(load_mask_3, value_3, value_4)
    selected_value_2_3_4 = tl.where(load_mask_2, value_2, selected_value_3_4)
    selected_value_1_2_3_4 = tl.where(load_mask_1, value_1, selected_value_2_3_4)
    final_value = tl.where(load_mask_0, value_0, selected_value_1_2_3_4)

    # Store the result
    tl.store(out_ptr0 + (linear_index), final_value, xmask)