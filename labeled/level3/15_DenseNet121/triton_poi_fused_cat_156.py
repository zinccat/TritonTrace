# From: 15_DenseNet121

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_156poi_fused_cat_156(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 297920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x_channel = (xindex // 49) % 608
    x_row = xindex % 49
    x_depth = xindex // 29792
    x_flat_index = xindex

    tmp_channel = x_channel
    tl.full([1], 0, tl.int64)
    tmp_max_channel_1 = tl.full([1], 512, tl.int64)
    tmp_channel_less_than_max_1 = tmp_channel < tmp_max_channel_1
    tmp_value_1 = tl.load(in_ptr0 + (x_row + 49 * (x_channel) + 25088 * x_depth), tmp_channel_less_than_max_1 & xmask, other=0.0)

    tmp_channel_greater_equal_max_1 = tmp_channel >= tmp_max_channel_1
    tmp_max_channel_2 = tl.full([1], 544, tl.int64)
    tmp_channel_less_than_max_2 = tmp_channel < tmp_max_channel_2
    tmp_condition_2 = tmp_channel_greater_equal_max_1 & tmp_channel_less_than_max_2
    tmp_value_2 = tl.load(in_ptr1 + (x_row + 49 * ((-512) + x_channel) + 1568 * x_depth), tmp_condition_2 & xmask, other=0.0)

    tmp_channel_greater_equal_max_2 = tmp_channel >= tmp_max_channel_2
    tmp_max_channel_3 = tl.full([1], 576, tl.int64)
    tmp_channel_less_than_max_3 = tmp_channel < tmp_max_channel_3
    tmp_condition_3 = tmp_channel_greater_equal_max_2 & tmp_channel_less_than_max_3
    tmp_value_3 = tl.load(in_ptr2 + (x_row + 49 * ((-544) + x_channel) + 1568 * x_depth), tmp_condition_3 & xmask, other=0.0)

    tmp_channel_greater_equal_max_3 = tmp_channel >= tmp_max_channel_3
    tmp_max_channel_4 = tl.full([1], 608, tl.int64)
    tmp_condition_4 = tmp_channel_greater_equal_max_3
    tmp_value_4 = tl.load(in_ptr3 + (x_row + 49 * ((-576) + x_channel) + 1568 * x_depth), tmp_condition_4 & xmask, other=0.0)

    tmp_value_3_or_4 = tl.where(tmp_condition_3, tmp_value_3, tmp_value_4)
    tmp_value_2_or_3_or_4 = tl.where(tmp_condition_2, tmp_value_2, tmp_value_3_or_4)
    tmp_final_value = tl.where(tmp_channel_less_than_max_1, tmp_value_1, tmp_value_2_or_3_or_4)

    tl.store(out_ptr0 + (x_flat_index), tmp_final_value, xmask)