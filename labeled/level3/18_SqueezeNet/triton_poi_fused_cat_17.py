# From: 18_SqueezeNet

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_17poi_fused_cat_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 186624
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x_mod_256 = xindex % 256
    x_div_256 = xindex // 256
    x_full_index = xindex

    zero_mask = tl.full([1], 0, tl.int64)
    threshold_mask = tl.full([1], 128, tl.int64)
    is_below_threshold = x_mod_256 < threshold_mask

    load0 = tl.load(in_ptr0 + (128 * x_div_256 + x_mod_256), is_below_threshold & xmask, eviction_policy='evict_last', other=0.0)
    load1 = tl.load(in_ptr1 + x_mod_256, is_below_threshold & xmask, eviction_policy='evict_last', other=0.0)
    sum_below_threshold = load0 + load1

    zero_int32 = tl.full([1], 0, tl.int32)
    max_below_threshold = triton_helpers.maximum(zero_int32, sum_below_threshold)
    zero_float = tl.full(max_below_threshold.shape, 0.0, max_below_threshold.dtype)
    result_below_threshold = tl.where(is_below_threshold, max_below_threshold, zero_float)

    is_above_threshold = x_mod_256 >= threshold_mask

    load2 = tl.load(in_ptr2 + (128 * x_div_256 + ((-128) + x_mod_256)), is_above_threshold & xmask, eviction_policy='evict_last', other=0.0)
    load3 = tl.load(in_ptr3 + ((-128) + x_mod_256), is_above_threshold & xmask, eviction_policy='evict_last', other=0.0)
    sum_above_threshold = load2 + load3

    max_above_threshold = triton_helpers.maximum(zero_int32, sum_above_threshold)
    result_above_threshold = tl.where(is_above_threshold, max_above_threshold, zero_float)

    final_result = tl.where(is_below_threshold, result_below_threshold, result_above_threshold)
    tl.store(out_ptr0 + x_full_index, final_result, xmask)