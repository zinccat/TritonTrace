# From: 14_Gemm_Divide_Sum_Scaling

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_div_mul_sum_0per_fused_div_mul_sum_0(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr):
    rnumel = 20
    RBLOCK: tl.constexpr = 32
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_indices = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_indices < rnumel
    row_indices = r_indices
    col_indices = x_indices
    loaded_values = tl.load(in_ptr0 + (row_indices + 20 * col_indices), r_mask & x_mask, other=0.0)
    scaling_factor = 0.5
    scaled_values = loaded_values * scaling_factor
    broadcasted_values = tl.broadcast_to(scaled_values, [XBLOCK, RBLOCK])
    masked_values = tl.where(r_mask & x_mask, broadcasted_values, 0)
    summed_values = tl.sum(masked_values, 1)[:, None]
    final_scaling_factor = 1.5
    final_result = summed_values * final_scaling_factor
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (col_indices), final_result, x_mask)