# From: 94_MSELoss

import triton
import triton.language as tl


@triton.jit
def triton_red_fused_mean_pow_sub_0(input_ptr0, input_ptr1, output_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    xnumel = 64
    rnumel = 8192
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_indices_0 = x_indices
    temp_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    
    for r_offset in range(0, rnumel, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < rnumel
        r_indices_1 = r_indices
        temp0 = tl.load(input_ptr0 + (r_indices_1 + (8192 * x_indices_0)), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        temp1 = tl.load(input_ptr1 + (r_indices_1 + (8192 * x_indices_0)), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        temp_diff = temp0 - temp1
        temp_squared = temp_diff * temp_diff
        temp_broadcast = tl.broadcast_to(temp_squared, [XBLOCK, RBLOCK])
        temp_accumulated = temp_sum + temp_broadcast
        temp_sum = tl.where(r_mask & x_mask, temp_accumulated, temp_sum)
    
    temp_result = tl.sum(temp_sum, 1)[:, None]
    tl.store(output_ptr0 + (x_indices_0), temp_result, x_mask)