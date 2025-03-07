# From: 38_L1Norm_

import triton
import triton.language as tl


@triton.jit
def triton_red_fused_abs_sum_0(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    xnumel = 32
    rnumel = 8192
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_indices_flat = x_indices
    temp_accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for r_offset in range(0, rnumel, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < rnumel
        r_indices_flat = r_indices
        loaded_values = tl.load(in_ptr0 + (r_indices_flat + (8192 * x_indices_flat)), r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        abs_values = tl.math.abs(loaded_values)
        broadcasted_values = tl.broadcast_to(abs_values, [XBLOCK, RBLOCK])
        temp_sum = temp_accumulator + broadcasted_values
        temp_accumulator = tl.where(r_mask & x_mask, temp_sum, temp_accumulator)
    
    reduced_sum = tl.sum(temp_accumulator, 1)[:, None]
    tl.store(out_ptr0 + (x_indices_flat), reduced_sum, x_mask)