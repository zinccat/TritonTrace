# From: 96_ConvTranspose3d_Multiply_Max_GlobalAvgPool_Clamp

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_red_fused_clamp_ge_le_logical_and_mean_1(in_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    rnumel = 14415
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_base = tl.arange(0, RBLOCK)[None, :]
    x0 = x_indices
    _accumulated_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for r_offset in range(0, rnumel, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < rnumel
        r1 = r_indices
        loaded_values = tl.load(in_ptr0 + (r1 + (14415 * x0)), r_mask, eviction_policy='evict_first', other=0.0)
        broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
        updated_sum = _accumulated_sum + broadcasted_values
        _accumulated_sum = tl.where(r_mask, updated_sum, _accumulated_sum)
    
    sum_per_x = tl.sum(_accumulated_sum, 1)[:, None]
    divisor = 14415.0
    mean_values = sum_per_x / divisor
    lower_bound = 0.0
    clamped_values = triton_helpers.maximum(mean_values, lower_bound)
    upper_bound = 1.0
    clamped_values = triton_helpers.minimum(clamped_values, upper_bound)
    
    greater_equal_mask = mean_values >= lower_bound
    less_equal_mask = mean_values <= upper_bound
    logical_and_mask = greater_equal_mask & less_equal_mask
    
    tl.store(out_ptr1 + (x0), clamped_values, None)
    tl.store(out_ptr2 + (x0), logical_and_mask, None)