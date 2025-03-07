# From: 97_Matmul_BatchNorm_BiasAdd_Divide_Swish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_div_mul_sigmoid_sigmoid_backward_sum_2(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 8
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    mask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_indices = tl.arange(0, RBLOCK)[None, :]
    mask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    row_indices = r_indices
    input_values = tl.load(in_ptr0 + (row_indices), None)
    broadcasted_values = tl.broadcast_to(input_values, [XBLOCK, RBLOCK])
    summed_values = tl.sum(broadcasted_values, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), summed_values, None)