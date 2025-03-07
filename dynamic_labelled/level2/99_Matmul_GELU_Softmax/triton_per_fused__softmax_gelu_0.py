# From: 99_Matmul_GELU_Softmax

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__softmax_gelu_0per_fused__softmax_gelu_0(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK: tl.constexpr):
    rnumel = 10
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    x_indices = xoffset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_indices = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_indices < rnumel
    row_indices = r_indices
    col_indices = x_indices
    input_values = tl.load(in_ptr0 + (row_indices + 10 * col_indices), r_mask & x_mask, other=0.0)
    half = 0.5
    scaled_input = input_values * half
    sqrt_half = 0.7071067811865476
    sqrt_half_scaled_input = input_values * sqrt_half
    erf_result = tl.extra.cuda.libdevice.erf(sqrt_half_scaled_input)
    one = 1.0
    erf_plus_one = erf_result + one
    gelu_result = scaled_input * erf_plus_one
    broadcast_gelu = tl.broadcast_to(gelu_result, [XBLOCK, RBLOCK])
    masked_gelu = tl.where(r_mask & x_mask, broadcast_gelu, float("-inf"))
    max_gelu = triton_helpers.max2(masked_gelu, 1)[:, None]
    softmax_input = gelu_result - max_gelu
    exp_values = tl.math.exp(softmax_input)
    broadcast_exp = tl.broadcast_to(exp_values, [XBLOCK, RBLOCK])
    masked_exp = tl.where(r_mask & x_mask, broadcast_exp, 0)
    sum_exp = tl.sum(masked_exp, 1)[:, None]
    softmax_output = exp_values / sum_exp
    tl.store(out_ptr2 + (row_indices + 10 * col_indices), softmax_output, r_mask & x_mask)