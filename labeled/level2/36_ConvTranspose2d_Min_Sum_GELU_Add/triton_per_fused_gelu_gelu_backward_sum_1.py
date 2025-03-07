# From: 36_ConvTranspose2d_Min_Sum_GELU_Add

import triton
import triton.language as tl


@triton.jit
def triton_per_fused_gelu_gelu_backward_sum_1(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 64
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_indices = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = r_indices
    x0 = x_indices % 64
    x1 = (x_indices // 64)
    x3 = x_indices
    input_value = tl.load(in_ptr0 + (x0 + (64 * r2) + (4096 * x1)), None)
    broadcasted_input = tl.broadcast_to(input_value, [XBLOCK, RBLOCK])
    sum_result = tl.sum(broadcasted_input, 1)[:, None]
    sqrt_half = 0.7071067811865476
    scaled_sum = sum_result * sqrt_half
    erf_result = tl.extra.cuda.libdevice.erf(scaled_sum)
    one = 1.0
    erf_plus_one = erf_result + one
    half = 0.5
    erf_half = erf_plus_one * half
    sum_squared = sum_result * sum_result
    neg_half = -0.5
    exp_argument = sum_squared * neg_half
    exp_result = tl.math.exp(exp_argument)
    inv_sqrt_2pi = 0.3989422804014327
    gaussian_term = exp_result * inv_sqrt_2pi
    scaled_gaussian = sum_result * gaussian_term
    final_result = erf_half + scaled_gaussian
    tl.store(out_ptr1 + (x3), final_result, None)
    tl.store(out_ptr0 + (x3), sum_result, None)