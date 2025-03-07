# From: 15_DenseNet121

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
    row_index = (xindex // 784) % 224
    col_index = xindex % 784
    channel_index = xindex // 175616
    linear_index = xindex
    tmp_row_index = row_index
    tl.full([1], 0, tl.int64)
    threshold_128 = tl.full([1], 128, tl.int64)
    is_below_128 = tmp_row_index < threshold_128
    value_below_128 = tl.load(in_ptr0 + (col_index + 784 * row_index + 100352 * channel_index), is_below_128 & xmask, other=0.0)
    
    threshold_160 = tl.full([1], 160, tl.int64)
    is_between_128_and_160 = (tmp_row_index >= threshold_128) & (tmp_row_index < threshold_160)
    value_between_128_and_160 = tl.load(in_ptr1 + (col_index + 784 * (row_index - 128) + 25088 * channel_index), is_between_128_and_160 & xmask, other=0.0)
    
    threshold_192 = tl.full([1], 192, tl.int64)
    is_between_160_and_192 = (tmp_row_index >= threshold_160) & (tmp_row_index < threshold_192)
    value_between_160_and_192 = tl.load(in_ptr2 + (col_index + 784 * (row_index - 160) + 25088 * channel_index), is_between_160_and_192 & xmask, other=0.0)
    
    is_above_192 = tmp_row_index >= threshold_192
    value_above_192 = tl.load(in_ptr3 + (col_index + 784 * (row_index - 192) + 25088 * channel_index), is_above_192 & xmask, other=0.0)
    
    value_between_160_and_192_or_above = tl.where(is_between_160_and_192, value_between_160_and_192, value_above_192)
    value_between_128_and_160_or_above = tl.where(is_between_128_and_160, value_between_128_and_160, value_between_160_and_192_or_above)
    final_value = tl.where(is_below_128, value_below_128, value_between_128_and_160_or_above)
    
    tl.store(out_ptr0 + (linear_index), final_value, xmask)