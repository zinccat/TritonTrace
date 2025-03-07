# From: 96_HuberLoss

import triton
import triton.language as tl


@triton.jit
def triton_per_fused_smooth_l1_loss_1(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 64
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    mask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_indices = tl.arange(0, RBLOCK)[None, :]
    mask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = r_indices
    input_values = tl.load(in_ptr0 + (r0), None)
    broadcasted_values = tl.broadcast_to(input_values, [XBLOCK, RBLOCK])
    sum_values = tl.sum(broadcasted_values, 1)[:, None]
    divisor = 524288.0
    result = sum_values / divisor
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), result, None)