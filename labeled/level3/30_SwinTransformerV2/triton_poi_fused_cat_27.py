# From: 30_SwinTransformerV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_27poi_fused_cat_27(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tl.full([1], 0, tl.int64)
    first_threshold = tl.full([1], 192, tl.int64)
    is_below_first_threshold = tmp0 < first_threshold
    data_from_first_input = tl.load(in_ptr0 + (x0), is_below_first_threshold & xmask, eviction_policy='evict_last', other=0.0)
    is_above_first_threshold = tmp0 >= first_threshold
    second_threshold = tl.full([1], 384, tl.int64)
    is_below_second_threshold = tmp0 < second_threshold
    is_between_thresholds = is_above_first_threshold & is_below_second_threshold
    zero_value = tl.full([1], 0.0, tl.float32)
    zero_tensor = tl.full(zero_value.shape, 0.0, zero_value.dtype)
    data_between_thresholds = tl.where(is_between_thresholds, zero_value, zero_tensor)
    is_above_second_threshold = tmp0 >= second_threshold
    third_threshold = tl.full([1], 576, tl.int64)
    data_from_second_input = tl.load(in_ptr1 + ((-384) + x0), is_above_second_threshold & xmask, eviction_policy='evict_last', other=0.0)
    data_from_second_or_between = tl.where(is_between_thresholds, data_between_thresholds, data_from_second_input)
    final_data = tl.where(is_below_first_threshold, data_from_first_input, data_from_second_or_between)
    tl.store(out_ptr0 + (x0), final_data, xmask)