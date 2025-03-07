# From: 16_DenseNet201

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_243poi_fused_cat_243(
    input_ptr0, input_ptr1, input_ptr2, input_ptr3, input_ptr4, input_ptr5, input_ptr6, 
    output_ptr0, xnumel, XBLOCK: tl.constexpr
):
    xnumel = 533120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel

    # Calculate indices
    channel_index = (xindex // 49) % 1088
    spatial_index = xindex % 49
    batch_index = xindex // 53312
    linear_index = xindex

    # Temporary variables for conditions
    tmp_channel_index = channel_index
    zero_value = tl.full([1], 0, tl.int64)
    threshold_896 = tl.full([1], 896, tl.int64)
    threshold_928 = tl.full([1], 928, tl.int64)
    threshold_960 = tl.full([1], 960, tl.int64)
    threshold_992 = tl.full([1], 992, tl.int64)
    threshold_1024 = tl.full([1], 1024, tl.int64)
    threshold_1056 = tl.full([1], 1056, tl.int64)
    threshold_1088 = tl.full([1], 1088, tl.int64)

    # Load and conditionally select values
    load_mask_0 = (channel_index < threshold_896) & xmask
    value_0 = tl.load(input_ptr0 + (spatial_index + 49 * channel_index + 43904 * batch_index), load_mask_0, other=0.0)

    load_mask_1 = (channel_index >= threshold_896) & (channel_index < threshold_928) & xmask
    value_1 = tl.load(input_ptr1 + (spatial_index + 49 * (channel_index - 896) + 1568 * batch_index), load_mask_1, other=0.0)

    load_mask_2 = (channel_index >= threshold_928) & (channel_index < threshold_960) & xmask
    value_2 = tl.load(input_ptr2 + (spatial_index + 49 * (channel_index - 928) + 1568 * batch_index), load_mask_2, other=0.0)

    load_mask_3 = (channel_index >= threshold_960) & (channel_index < threshold_992) & xmask
    value_3 = tl.load(input_ptr3 + (spatial_index + 49 * (channel_index - 960) + 1568 * batch_index), load_mask_3, other=0.0)

    load_mask_4 = (channel_index >= threshold_992) & (channel_index < threshold_1024) & xmask
    value_4 = tl.load(input_ptr4 + (spatial_index + 49 * (channel_index - 992) + 1568 * batch_index), load_mask_4, other=0.0)

    load_mask_5 = (channel_index >= threshold_1024) & (channel_index < threshold_1056) & xmask
    value_5 = tl.load(input_ptr5 + (spatial_index + 49 * (channel_index - 1024) + 1568 * batch_index), load_mask_5, other=0.0)

    load_mask_6 = (channel_index >= threshold_1056) & xmask
    value_6 = tl.load(input_ptr6 + (spatial_index + 49 * (channel_index - 1056) + 1568 * batch_index), load_mask_6, other=0.0)

    # Conditional selection
    selected_value = tl.where(load_mask_5, value_5, value_6)
    selected_value = tl.where(load_mask_4, value_4, selected_value)
    selected_value = tl.where(load_mask_3, value_3, selected_value)
    selected_value = tl.where(load_mask_2, value_2, selected_value)
    selected_value = tl.where(load_mask_1, value_1, selected_value)
    selected_value = tl.where(load_mask_0, value_0, selected_value)

    # Store the result
    tl.store(output_ptr0 + (linear_index), selected_value, xmask)