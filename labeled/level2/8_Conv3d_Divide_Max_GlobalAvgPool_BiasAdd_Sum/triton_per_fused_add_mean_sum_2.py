# From: 8_Conv3d_Divide_Max_GlobalAvgPool_BiasAdd_Sum

import triton
import triton.language as tl


@triton.jit
def triton_per_fused_add_mean_sum_2(input_ptr0, input_ptr1, output_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    x_indices = xoffset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_indices = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = r_indices
    x0 = x_indices
    loaded_values0 = tl.load(input_ptr0 + (r1 + (16 * x0)), x_mask, other=0.0)
    loaded_values1 = tl.load(input_ptr1 + (r1), None, eviction_policy='evict_last')
    divisor = 1575.0
    normalized_values = loaded_values0 / divisor
    combined_values = normalized_values + loaded_values1
    broadcasted_values = tl.broadcast_to(combined_values, [XBLOCK, RBLOCK])
    masked_values = tl.where(x_mask, broadcasted_values, 0)
    summed_values = tl.sum(masked_values, 1)[:, None]
    tl.store(output_ptr0 + (x0), summed_values, x_mask)