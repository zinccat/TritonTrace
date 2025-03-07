# From: 28_VisionTransformer

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_cat_2poi_fused_add_cat_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 201728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x_row = (xindex // 512) % 197
    x_col = xindex % 512
    x_depth = xindex // 100864
    x_mod_depth = xindex % 100864
    x_flat_index = xindex

    tmp_value_from_in_ptr2 = tl.load(in_ptr2 + (x_mod_depth), xmask, eviction_policy='evict_last')
    tmp_row_check = x_row
    tl.full([1], 0, tl.int64)
    tmp_row_limit = tl.full([1], 1, tl.int64)
    is_within_row_limit = tmp_row_check < tmp_row_limit

    tmp_value_from_in_ptr0 = tl.load(in_ptr0 + (x_col), is_within_row_limit & xmask, eviction_policy='evict_last', other=0.0)
    is_exceeding_row_limit = tmp_row_check >= tmp_row_limit
    tl.full([1], 197, tl.int64)
    tmp_value_from_in_ptr1 = tl.load(in_ptr1 + (x_col + 512 * ((-1) + x_row) + 100352 * x_depth), is_exceeding_row_limit & xmask, other=0.0)

    tmp_combined_value = tl.where(is_within_row_limit, tmp_value_from_in_ptr0, tmp_value_from_in_ptr1)
    tmp_result = tmp_combined_value + tmp_value_from_in_ptr2

    tl.store(out_ptr0 + (x_flat_index), tmp_result, xmask)