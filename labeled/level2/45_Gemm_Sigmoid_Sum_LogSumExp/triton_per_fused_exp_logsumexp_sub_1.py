# From: 45_Gemm_Sigmoid_Sum_LogSumExp

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_per_fused_exp_logsumexp_sub_1(in_out_ptr0, in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    x_indices = xoffset + tl.arange(0, XBLOCK)[:, None]
    r_indices = tl.arange(0, RBLOCK)[None, :]
    
    # Load input data
    input_data = tl.load(in_ptr0 + (r_indices), None)
    broadcast_input = tl.broadcast_to(input_data, [XBLOCK, RBLOCK])
    
    # Compute max for numerical stability
    max_values = triton_helpers.max2(broadcast_input, 1)[:, None]
    abs_max_values = tl.math.abs(max_values)
    inf_value = float("inf")
    is_inf = abs_max_values == inf_value
    zero_value = 0.0
    stable_max = tl.where(is_inf, zero_value, max_values)
    
    # Compute exponentials
    shifted_input = input_data - stable_max
    exp_shifted_input = tl.math.exp(shifted_input)
    broadcast_exp = tl.broadcast_to(exp_shifted_input, [XBLOCK, RBLOCK])
    
    # Compute log-sum-exp
    sum_exp = tl.sum(broadcast_exp, 1)[:, None]
    log_sum_exp = tl.math.log(sum_exp)
    log_sum_exp_stable = log_sum_exp + stable_max
    
    # Compute final output
    final_shifted_input = input_data - log_sum_exp_stable
    final_output = tl.math.exp(final_shifted_input)
    
    # Store results
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), log_sum_exp_stable, None)
    tl.store(out_ptr1 + (tl.broadcast_to(r_indices, [XBLOCK, RBLOCK])), final_output, None)