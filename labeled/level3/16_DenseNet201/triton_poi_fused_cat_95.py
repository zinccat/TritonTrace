# From: 16_DenseNet201

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_95poi_fused_cat_95(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 689920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel

    # Calculate indices
    channel_index = (xindex // 196) % 352
    row_index = xindex % 196
    batch_index = xindex // 68992
    linear_index = xindex

    # Temporary variables for conditions
    channel_limit_1 = 256
    channel_limit_2 = 288
    channel_limit_3 = 320

    # Load data based on conditions
    load_mask_1 = channel_index < channel_limit_1
    data_1 = tl.load(in_ptr0 + (row_index + 196 * channel_index + 50176 * batch_index), load_mask_1 & xmask, other=0.0)

    load_mask_2 = (channel_index >= channel_limit_1) & (channel_index < channel_limit_2)
    data_2 = tl.load(in_ptr1 + (row_index + 196 * ((-channel_limit_1) + channel_index) + 6272 * batch_index), load_mask_2 & xmask, other=0.0)

    load_mask_3 = (channel_index >= channel_limit_2) & (channel_index < channel_limit_3)
    data_3 = tl.load(in_ptr2 + (row_index + 196 * ((-channel_limit_2) + channel_index) + 6272 * batch_index), load_mask_3 & xmask, other=0.0)

    load_mask_4 = channel_index >= channel_limit_3
    data_4 = tl.load(in_ptr3 + (row_index + 196 * ((-channel_limit_3) + channel_index) + 6272 * batch_index), load_mask_4 & xmask, other=0.0)

    # Combine data based on conditions
    combined_data_3_4 = tl.where(load_mask_3, data_3, data_4)
    combined_data_2_34 = tl.where(load_mask_2, data_2, combined_data_3_4)
    final_data = tl.where(load_mask_1, data_1, combined_data_2_34)

    # Store the result
    tl.store(out_ptr0 + (linear_index), final_data, xmask)