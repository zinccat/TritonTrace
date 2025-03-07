# From: 30_SwinTransformerV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_50poi_fused_cat_50(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tl.full([1], 0, tl.int64)
    first_segment_size = tl.full([1], 384, tl.int64)
    is_in_first_segment = tmp0 < first_segment_size
    load_first_segment = tl.load(in_ptr0 + (x0), is_in_first_segment & xmask, eviction_policy='evict_last', other=0.0)
    is_in_second_segment = (tmp0 >= first_segment_size) & (tmp0 < tl.full([1], 768, tl.int64))
    zero_value = tl.full([1], 0.0, tl.float32)
    second_segment_value = tl.where(is_in_second_segment, zero_value, zero_value)
    is_in_third_segment = tmp0 >= tl.full([1], 768, tl.int64)
    load_third_segment = tl.load(in_ptr1 + ((-768) + x0), is_in_third_segment & xmask, eviction_policy='evict_last', other=0.0)
    select_second_or_third_segment = tl.where(is_in_second_segment, second_segment_value, load_third_segment)
    select_first_or_others = tl.where(is_in_first_segment, load_first_segment, select_second_or_third_segment)
    tl.store(out_ptr0 + (x0), select_first_or_others, xmask)