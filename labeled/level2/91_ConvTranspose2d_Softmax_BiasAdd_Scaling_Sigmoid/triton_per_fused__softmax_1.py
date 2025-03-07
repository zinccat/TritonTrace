# From: 91_ConvTranspose2d_Softmax_BiasAdd_Scaling_Sigmoid

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_per_fused__softmax_1(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 139392
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    x_indices = xoffset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_indices = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = r_indices
    x_col = x_indices % 1089
    x_row = (x_indices // 1089)
    
    # Load input values with masking
    input_values = tl.load(in_ptr0 + (x_col + (1089 * r2) + (69696 * x_row)), x_mask, other=0.0)
    broadcasted_values = tl.broadcast_to(input_values, [XBLOCK, RBLOCK])
    masked_values = tl.where(x_mask, broadcasted_values, float("-inf"))
    
    # Compute max for numerical stability
    max_values = triton_helpers.max2(masked_values, 1)[:, None]
    shifted_values = input_values - max_values
    
    # Exponentiate and sum for softmax
    exp_values = tl.math.exp(shifted_values)
    broadcasted_exp_values = tl.broadcast_to(exp_values, [XBLOCK, RBLOCK])
    masked_exp_values = tl.where(x_mask, broadcasted_exp_values, 0)
    sum_exp_values = tl.sum(masked_exp_values, 1)[:, None]
    
    # Store results
    tl.store(out_ptr0 + (x_col + (1120 * x_row)), max_values, x_mask)
    tl.store(out_ptr1 + (x_col + (1120 * x_row)), sum_exp_values, x_mask)