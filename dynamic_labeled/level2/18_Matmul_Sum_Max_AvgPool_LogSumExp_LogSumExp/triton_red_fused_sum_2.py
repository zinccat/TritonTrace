# From: 18_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_sum_2(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    xnumel = 5
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x_indices_copy = x_indices

    for r_offset in range(0, rnumel, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < rnumel
        r_indices_copy = r_indices
        loaded_values = tl.load(in_ptr0 + (r_indices_copy), r_mask, eviction_policy='evict_last', other=0.0)
        broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
        temp_sum = temp_accumulator + broadcasted_values
        temp_accumulator = tl.where(r_mask & x_mask, temp_sum, temp_accumulator)

    summed_values = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(out_ptr0 + (x_indices_copy), summed_values, x_mask)