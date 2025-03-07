# From: 18_SqueezeNet

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_cat_20poi_fused_cat_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 279936
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x_mod_384 = xindex % 384
    x_div_384 = xindex // 384
    x_full_index = xindex

    zero_int64 = tl.full([1], 0, tl.int64)
    half_384 = tl.full([1], 192, tl.int64)
    is_first_half = x_mod_384 < half_384

    load_first_half_0 = tl.load(in_ptr0 + (half_384 * x_div_384 + x_mod_384), is_first_half & xmask, eviction_policy='evict_last', other=0.0)
    load_first_half_1 = tl.load(in_ptr1 + x_mod_384, is_first_half & xmask, eviction_policy='evict_last', other=0.0)
    sum_first_half = load_first_half_0 + load_first_half_1

    zero_int32 = tl.full([1], 0, tl.int32)
    max_first_half = triton_helpers.maximum(zero_int32, sum_first_half)
    zero_float = tl.full(max_first_half.shape, 0.0, max_first_half.dtype)
    result_first_half = tl.where(is_first_half, max_first_half, zero_float)

    is_second_half = x_mod_384 >= half_384

    load_second_half_0 = tl.load(in_ptr2 + (half_384 * x_div_384 + (x_mod_384 - half_384)), is_second_half & xmask, eviction_policy='evict_last', other=0.0)
    load_second_half_1 = tl.load(in_ptr3 + (x_mod_384 - half_384), is_second_half & xmask, eviction_policy='evict_last', other=0.0)
    sum_second_half = load_second_half_0 + load_second_half_1

    max_second_half = triton_helpers.maximum(zero_int32, sum_second_half)
    result_second_half = tl.where(is_second_half, max_second_half, zero_float)

    final_result = tl.where(is_first_half, result_first_half, result_second_half)
    tl.store(out_ptr0 + x_full_index, final_result, xmask)