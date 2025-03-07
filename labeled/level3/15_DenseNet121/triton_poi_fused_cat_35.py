# From: 15_DenseNet121

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_35poi_fused_cat_35(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1254400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    batch_index = (xindex // 784) % 160
    pixel_index = xindex % 784
    channel_index = xindex // 125440
    linear_index = xindex

    tmp0 = batch_index
    tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 128, tl.int64)
    is_first_half = tmp0 < tmp3
    value_from_first_input = tl.load(in_ptr0 + (pixel_index + 784 * batch_index + 100352 * channel_index), is_first_half & xmask, other=0.0)
    is_second_half = tmp0 >= tmp3
    tl.full([1], 160, tl.int64)
    value_from_second_input = tl.load(in_ptr1 + (pixel_index + 784 * ((-128) + batch_index) + 25088 * channel_index), is_second_half & xmask, other=0.0)
    selected_value = tl.where(is_first_half, value_from_first_input, value_from_second_input)
    tl.store(out_ptr0 + (linear_index), selected_value, xmask)