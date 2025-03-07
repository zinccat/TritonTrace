# From: 99_Matmul_GELU_Softmax

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__softmax_backward_data_gelu_gelu_backward_0(
    in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK: tl.constexpr
):
    rnumel = 10
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    x_indices = xoffset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_indices = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_indices < rnumel
    row_indices = r_indices
    col_indices = x_indices

    input_data0 = tl.load(in_ptr0 + (row_indices + 10 * col_indices), r_mask & x_mask, other=0.0)
    input_data1 = tl.load(in_ptr1 + (row_indices + 10 * col_indices), r_mask & x_mask, other=0.0)
    output_data = tl.load(in_out_ptr0 + (row_indices + 10 * col_indices), r_mask & x_mask, other=0.0)

    elementwise_product = input_data0 * input_data1
    broadcasted_product = tl.broadcast_to(elementwise_product, [XBLOCK, RBLOCK])
    masked_broadcast = tl.where(r_mask & x_mask, broadcasted_product, 0)
    sum_over_rows = tl.sum(masked_broadcast, 1)[:, None]

    neg_input_data1 = -input_data1
    fused_multiply_add = tl.extra.cuda.libdevice.fma(neg_input_data1, sum_over_rows, elementwise_product)

    sqrt_half = 0.7071067811865476
    scaled_output_data = output_data * sqrt_half
    erf_result = tl.extra.cuda.libdevice.erf(scaled_output_data)
    one = 1.0
    erf_plus_one = erf_result + one
    half = 0.5
    erf_half = erf_plus_one * half

    squared_output_data = output_data * output_data
    neg_half = -0.5
    exp_argument = squared_output_data * neg_half
    exp_result = tl.math.exp(exp_argument)
    sqrt_two_pi = 0.3989422804014327
    gaussian_term = exp_result * sqrt_two_pi
    output_data_times_gaussian = output_data * gaussian_term
    gelu_result = erf_half + output_data_times_gaussian

    final_result = fused_multiply_add * gelu_result
    tl.store(in_out_ptr0 + (row_indices + 10 * col_indices), final_result, r_mask & x_mask)