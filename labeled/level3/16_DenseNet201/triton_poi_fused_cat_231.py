# From: 16_DenseNet201

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_231poi_fused_cat_231(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 470400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel

    # Calculate indices for accessing input tensors
    channel_index = (xindex // 49) % 960
    row_index = xindex % 49
    batch_index = xindex // 47040
    flat_index = xindex

    # Temporary variables for conditions
    channel_limit_1 = 896
    channel_limit_2 = 928

    # Load data from input pointers with masking
    load_mask_1 = channel_index < channel_limit_1
    data_from_in_ptr0 = tl.load(in_ptr0 + (row_index + 49 * channel_index + 43904 * batch_index), load_mask_1 & xmask, other=0.0)

    load_mask_2 = (channel_index >= channel_limit_1) & (channel_index < channel_limit_2)
    data_from_in_ptr1 = tl.load(in_ptr1 + (row_index + 49 * ((-channel_limit_1) + channel_index) + 1568 * batch_index), load_mask_2 & xmask, other=0.0)

    load_mask_3 = channel_index >= channel_limit_2
    data_from_in_ptr2 = tl.load(in_ptr2 + (row_index + 49 * ((-channel_limit_2) + channel_index) + 1568 * batch_index), load_mask_3 & xmask, other=0.0)

    # Select data based on conditions
    selected_data_1 = tl.where(load_mask_2, data_from_in_ptr1, data_from_in_ptr2)
    final_selected_data = tl.where(load_mask_1, data_from_in_ptr0, selected_data_1)

    # Store the result in the output pointer
    tl.store(out_ptr0 + (flat_index), final_selected_data, xmask)