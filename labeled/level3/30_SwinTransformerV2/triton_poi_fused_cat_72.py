# From: 30_SwinTransformerV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_72poi_fused_cat_72(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tl.full([1], 0, tl.int64)
    first_split_point = tl.full([1], 768, tl.int64)
    is_before_first_split = tmp0 < first_split_point
    data_from_first_input = tl.load(in_ptr0 + (x0), is_before_first_split & xmask, eviction_policy='evict_last', other=0.0)
    is_after_first_split = tmp0 >= first_split_point
    second_split_point = tl.full([1], 1536, tl.int64)
    is_before_second_split = tmp0 < second_split_point
    is_between_splits = is_after_first_split & is_before_second_split
    zero_value = tl.full([1], 0.0, tl.float32)
    zero_tensor = tl.full(zero_value.shape, 0.0, zero_value.dtype)
    data_between_splits = tl.where(is_between_splits, zero_value, zero_tensor)
    is_after_second_split = tmp0 >= second_split_point
    full_xnumel = tl.full([1], 2304, tl.int64)
    data_from_second_input = tl.load(in_ptr1 + ((-1536) + x0), is_after_second_split & xmask, eviction_policy='evict_last', other=0.0)
    data_from_second_or_between = tl.where(is_between_splits, data_between_splits, data_from_second_input)
    final_data = tl.where(is_before_first_split, data_from_first_input, data_from_second_or_between)
    tl.store(out_ptr0 + (x0), final_data, xmask)