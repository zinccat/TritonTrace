# From: 15_DenseNet121

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_150poi_fused_cat_150(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 266560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    channel_index = (xindex // 49) % 544
    spatial_index = xindex % 49
    batch_index = xindex // 26656
    linear_index = xindex

    tmp_channel_index = channel_index
    tl.full([1], 0, tl.int64)
    max_channel_index = tl.full([1], 512, tl.int64)
    is_within_first_input = tmp_channel_index < max_channel_index

    load_from_first_input = tl.load(
        in_ptr0 + (spatial_index + 49 * channel_index + 25088 * batch_index),
        is_within_first_input & xmask,
        other=0.0
    )

    is_within_second_input = tmp_channel_index >= max_channel_index
    second_input_offset = tl.full([1], 544, tl.int64)

    load_from_second_input = tl.load(
        in_ptr1 + (spatial_index + 49 * ((-512) + channel_index) + 1568 * batch_index),
        is_within_second_input & xmask,
        other=0.0
    )

    selected_value = tl.where(is_within_first_input, load_from_first_input, load_from_second_input)
    tl.store(out_ptr0 + (linear_index), selected_value, xmask)