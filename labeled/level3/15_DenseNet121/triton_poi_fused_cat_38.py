# From: 15_DenseNet121

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_38poi_fused_cat_38(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1505280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    channel_index = (xindex // 784) % 192
    pixel_index = xindex % 784
    batch_index = xindex // 150528
    linear_index = xindex
    tmp_channel_index = channel_index
    tl.full([1], 0, tl.int64)
    threshold_128 = tl.full([1], 128, tl.int64)
    is_less_than_128 = tmp_channel_index < threshold_128
    value_from_in_ptr0 = tl.load(in_ptr0 + (pixel_index + 784 * channel_index + 100352 * batch_index), is_less_than_128 & xmask, other=0.0)
    is_greater_equal_128 = tmp_channel_index >= threshold_128
    threshold_160 = tl.full([1], 160, tl.int64)
    is_less_than_160 = tmp_channel_index < threshold_160
    is_between_128_and_160 = is_greater_equal_128 & is_less_than_160
    value_from_in_ptr1 = tl.load(in_ptr1 + (pixel_index + 784 * ((-128) + channel_index) + 25088 * batch_index), is_between_128_and_160 & xmask, other=0.0)
    is_greater_equal_160 = tmp_channel_index >= threshold_160
    threshold_192 = tl.full([1], 192, tl.int64)
    value_from_in_ptr2 = tl.load(in_ptr2 + (pixel_index + 784 * ((-160) + channel_index) + 25088 * batch_index), is_greater_equal_160 & xmask, other=0.0)
    value_from_in_ptr1_or_in_ptr2 = tl.where(is_between_128_and_160, value_from_in_ptr1, value_from_in_ptr2)
    final_value = tl.where(is_less_than_128, value_from_in_ptr0, value_from_in_ptr1_or_in_ptr2)
    tl.store(out_ptr0 + (linear_index), final_value, xmask)