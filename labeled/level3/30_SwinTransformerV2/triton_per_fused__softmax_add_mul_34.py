# From: 30_SwinTransformerV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__softmax_add_mul_34per_fused__softmax_add_mul_34(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK: tl.constexpr
):
    xnumel = 47040
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_index = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_index < rnumel
    row_index = r_index
    col_index = x_index
    block_row_index = x_index // 49
    col_within_block = x_index % 49
    block_col_index = x_index // 49

    tmp0 = tl.load(in_ptr0 + (row_index + 49 * col_index), r_mask & x_mask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (block_row_index % 6), x_mask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (row_index + 49 * col_within_block), r_mask & x_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.full([XBLOCK, RBLOCK], 169, tl.int32)
    tmp5 = tmp3 + tmp4
    tmp6 = tmp3 < 0
    tmp7 = tl.where(tmp6, tmp5, tmp3)
    tl.device_assert(((0 <= tmp7) & (tmp7 < 169)) | ~(r_mask & x_mask), "index out of bounds: 0 <= tmp7 < 169")
    tmp9 = tl.load(in_ptr3 + (block_row_index % 6 + 6 * tmp7), r_mask & x_mask, eviction_policy='evict_last')
    tmp10 = tl.sigmoid(tmp9)
    tmp11 = 16.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp2 + tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp16 = tl.where(r_mask & x_mask, tmp14, float("-inf"))
    tmp17 = triton_helpers.max2(tmp16, 1)[:, None]
    tmp18 = tmp13 - tmp17
    tmp19 = tl.math.exp(tmp18)
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
    tmp22 = tl.where(r_mask & x_mask, tmp20, 0)
    tmp23 = tl.sum(tmp22, 1)[:, None]
    tmp24 = tmp19 / tmp23
    tl.store(in_out_ptr0 + (row_index + 49 * col_within_block + 2432 * block_col_index), tmp24, r_mask & x_mask)