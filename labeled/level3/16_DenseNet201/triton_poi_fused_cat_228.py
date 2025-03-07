# From: 16_DenseNet201

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_228poi_fused_cat_228(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 454720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    channel_index = (xindex // 49) % 928
    row_index = xindex % 49
    batch_index = xindex // 45472
    linear_index = xindex

    tmp_channel_index = channel_index
    tl.full([1], 0, tl.int64)
    max_channel_index = tl.full([1], 896, tl.int64)
    is_within_first_input = tmp_channel_index < max_channel_index

    value_from_first_input = tl.load(
        in_ptr0 + (row_index + 49 * channel_index + 43904 * batch_index),
        is_within_first_input & xmask,
        other=0.0
    )

    is_out_of_bounds_first_input = tmp_channel_index >= max_channel_index
    max_channel_index_full = tl.full([1], 928, tl.int64)

    value_from_second_input = tl.load(
        in_ptr1 + (row_index + 49 * ((-896) + channel_index) + 1568 * batch_index),
        is_out_of_bounds_first_input & xmask,
        other=0.0
    )

    selected_value = tl.where(is_within_first_input, value_from_first_input, value_from_second_input)
    tl.store(out_ptr0 + (linear_index), selected_value, xmask)