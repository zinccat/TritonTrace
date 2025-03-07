# From: 47_Sum_reduction_over_a_dimension

import triton
import triton.language as tl


@triton.jit
def triton_red_fused_sum_0(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    rnumel = 128
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_mod_256 = x_indices % 256
    x_div_256 = (x_indices // 256)
    temp_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x_full_indices = x_indices

    for r_offset in range(0, rnumel, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < rnumel
        r_indices_adjusted = r_indices
        loaded_values = tl.load(in_ptr0 + (x_mod_256 + (256 * r_indices_adjusted) + (32768 * x_div_256)), r_mask, eviction_policy='evict_first', other=0.0)
        broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
        temp_sum_update = temp_sum + broadcasted_values
        temp_sum = tl.where(r_mask, temp_sum_update, temp_sum)

    reduced_sum = tl.sum(temp_sum, 1)[:, None]
    tl.store(out_ptr0 + (x_full_indices), reduced_sum, None)