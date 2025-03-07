# From: 8_Conv3d_Divide_Max_GlobalAvgPool_BiasAdd_Sum

import triton
import triton.language as tl


@triton.jit
def triton_red_fused_mean_1(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    rnumel = 1575
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_indices_0 = x_indices
    _accumulated_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for r_offset in range(0, rnumel, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < rnumel
        r_indices_1 = r_indices
        loaded_values = tl.load(in_ptr0 + (r_indices_1 + (1575 * x_indices_0)), r_mask, eviction_policy='evict_first', other=0.0)
        broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
        updated_sum = _accumulated_sum + broadcasted_values
        _accumulated_sum = tl.where(r_mask, updated_sum, _accumulated_sum)
    
    summed_values = tl.sum(_accumulated_sum, 1)[:, None]
    tl.store(out_ptr0 + (x_indices_0), summed_values, None)