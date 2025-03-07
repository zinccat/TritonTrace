# From: 46_NetVladWithGhostClusters

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_sum_4red_fused_sum_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    xnumel = 1024
    rnumel = 100
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_mod_32 = x_index % 32
    x_div_32 = x_index // 32
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x_full_index = x_index

    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r_index_flat = r_index
        temp0 = tl.load(in_ptr0 + (x_mod_32 + 48 * r_index_flat + 4800 * x_div_32), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        temp1 = tl.load(in_ptr1 + (r_index_flat + 100 * x_div_32), r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        temp4 = tl.load(in_ptr2 + (r_index_flat + 100 * x_div_32), r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        temp2 = temp0 - temp1
        temp3 = tl.math.exp(temp2)
        temp5 = temp3 / temp4
        temp_broadcast = tl.broadcast_to(temp5, [XBLOCK, RBLOCK])
        temp_accumulator = temp_accumulator + temp_broadcast
        temp_accumulator = tl.where(r_mask & x_mask, temp_accumulator, temp_accumulator)

    temp_sum = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(out_ptr0 + (x_full_index), temp_sum, x_mask)