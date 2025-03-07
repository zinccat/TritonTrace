# From: 30_SwinTransformerV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__softmax_add_64(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 23520
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_index = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_index < rnumel
    row_index = r_index
    col_index = x_index
    block_index = (x_index // 49) % 12
    col_within_block = x_index % 49
    depth_index = (x_index // 588) % 4
    row_within_block = x_index // 49

    temp0 = tl.load(in_ptr0 + (row_index + 49 * col_index), r_mask & x_mask, other=0.0)
    temp1 = tl.load(in_ptr1 + (block_index), x_mask, eviction_policy='evict_last')
    temp3 = tl.load(in_ptr2 + (row_index + 49 * col_within_block), r_mask & x_mask, eviction_policy='evict_last', other=0.0)
    temp14 = tl.load(in_ptr4 + (row_index + 49 * col_within_block + 2401 * depth_index), r_mask & x_mask, eviction_policy='evict_last', other=0.0)

    temp2 = temp0 * temp1
    temp4 = tl.full([XBLOCK, RBLOCK], 169, tl.int32)
    temp5 = temp3 + temp4
    temp6 = temp3 < 0
    temp7 = tl.where(temp6, temp5, temp3)

    tl.device_assert(((0 <= temp7) & (temp7 < 169)) | ~(r_mask & x_mask), "index out of bounds: 0 <= temp7 < 169")

    temp9 = tl.load(in_ptr3 + (block_index + 12 * temp7), r_mask & x_mask, eviction_policy='evict_last')
    temp10 = tl.sigmoid(temp9)
    temp11 = 16.0
    temp12 = temp10 * temp11
    temp13 = temp2 + temp12
    temp15 = temp13 + temp14
    temp16 = tl.broadcast_to(temp15, [XBLOCK, RBLOCK])
    temp18 = tl.where(r_mask & x_mask, temp16, float("-inf"))
    temp19 = triton_helpers.max2(temp18, 1)[:, None]
    temp20 = temp15 - temp19
    temp21 = tl.math.exp(temp20)
    temp22 = tl.broadcast_to(temp21, [XBLOCK, RBLOCK])
    temp24 = tl.where(r_mask & x_mask, temp22, 0)
    temp25 = tl.sum(temp24, 1)[:, None]
    temp26 = temp21 / temp25

    tl.store(in_out_ptr0 + (row_index + 49 * col_within_block + 2432 * row_within_block), temp26, r_mask & x_mask)