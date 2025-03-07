# From: 16_DenseNet201

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_41poi_fused_cat_41(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1756160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel

    # Calculate indices
    channel_index = (xindex // 784) % 224
    row_index = xindex % 784
    batch_index = xindex // 175616
    flat_index = xindex

    # Temporary variables for conditions
    channel_threshold_128 = tl.full([1], 128, tl.int64)
    channel_threshold_160 = tl.full([1], 160, tl.int64)
    channel_threshold_192 = tl.full([1], 192, tl.int64)
    channel_threshold_224 = tl.full([1], 224, tl.int64)

    # Load and conditionally select values
    load_mask_0 = channel_index < channel_threshold_128
    value_0 = tl.load(in_ptr0 + (row_index + 784 * channel_index + 100352 * batch_index), load_mask_0 & xmask, other=0.0)

    load_mask_1 = (channel_index >= channel_threshold_128) & (channel_index < channel_threshold_160)
    value_1 = tl.load(in_ptr1 + (row_index + 784 * ((-128) + channel_index) + 25088 * batch_index), load_mask_1 & xmask, other=0.0)

    load_mask_2 = (channel_index >= channel_threshold_160) & (channel_index < channel_threshold_192)
    value_2 = tl.load(in_ptr2 + (row_index + 784 * ((-160) + channel_index) + 25088 * batch_index), load_mask_2 & xmask, other=0.0)

    load_mask_3 = channel_index >= channel_threshold_192
    value_3 = tl.load(in_ptr3 + (row_index + 784 * ((-192) + channel_index) + 25088 * batch_index), load_mask_3 & xmask, other=0.0)

    # Combine values based on conditions
    combined_value = tl.where(load_mask_2, value_2, value_3)
    combined_value = tl.where(load_mask_1, value_1, combined_value)
    combined_value = tl.where(load_mask_0, value_0, combined_value)

    # Store the result
    tl.store(out_ptr0 + (flat_index), combined_value, xmask)