# From: 58_ConvTranspose3d_LogSumExp_HardSwish_Subtract_Clamp_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_div_ge_le_logical_and_logsumexp_mul_neg_sigmoid_sub_sum_where_4per_fused_add_div_ge_le_logical_and_logsumexp_mul_neg_sigmoid_sub_sum_where_4(
    input_ptr, output_ptr, num_elements_x, num_elements_r, XBLOCK: tl.constexpr
):
    num_elements_x = 16
    num_elements_r = 123
    RBLOCK: tl.constexpr = 128

    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements_x

    r_indices = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_indices < num_elements_r

    r1 = r_indices
    x0 = x_indices

    tmp0 = tl.load(input_ptr + (x0 + 16 * r1), r_mask & x_mask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(r_mask & x_mask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]

    tl.store(output_ptr + (x0), tmp4, x_mask)