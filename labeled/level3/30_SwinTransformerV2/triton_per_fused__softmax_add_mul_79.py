# From: 30_SwinTransformerV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__softmax_add_mul_79per_fused__softmax_add_mul_79(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK: tl.constexpr
):
    xnumel = 11760
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_index = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_index < rnumel
    row_index = r_index
    col_index = x_index
    batch_index = (x_index // 49) % 24
    channel_index = x_index % 49
    block_index = x_index // 49
    temp0 = tl.load(in_ptr0 + (row_index + 49 * col_index), r_mask & x_mask, other=0.0)
    temp1 = tl.load(in_ptr1 + (batch_index), x_mask, eviction_policy='evict_last')
    temp3 = tl.load(in_ptr2 + (row_index + 49 * channel_index), r_mask & x_mask, eviction_policy='evict_last', other=0.0)
    temp2 = temp0 * temp1
    temp4 = tl.full([XBLOCK, RBLOCK], 169, tl.int32)
    temp5 = temp3 + temp4
    temp6 = temp3 < 0
    temp7 = tl.where(temp6, temp5, temp3)
    tl.device_assert(((0 <= temp7) & (temp7 < 169)) | ~(r_mask & x_mask), "index out of bounds: 0 <= temp7 < 169")
    temp9 = tl.load(in_ptr3 + (batch_index + 24 * temp7), r_mask & x_mask, eviction_policy='evict_last')
    temp10 = tl.sigmoid(temp9)
    temp11 = 16.0
    temp12 = temp10 * temp11
    temp13 = temp2 + temp12
    temp14 = tl.broadcast_to(temp13, [XBLOCK, RBLOCK])
    temp16 = tl.where(r_mask & x_mask, temp14, float("-inf"))
    temp17 = triton_helpers.max2(temp16, 1)[:, None]
    temp18 = temp13 - temp17
    temp19 = tl.math.exp(temp18)
    temp20 = tl.broadcast_to(temp19, [XBLOCK, RBLOCK])
    temp22 = tl.where(r_mask & x_mask, temp20, 0)
    temp23 = tl.sum(temp22, 1)[:, None]
    temp24 = temp19 / temp23
    tl.store(in_out_ptr0 + (row_index + 49 * channel_index + 2432 * block_index), temp24, r_mask & x_mask)