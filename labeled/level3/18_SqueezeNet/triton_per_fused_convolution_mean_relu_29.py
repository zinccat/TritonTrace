# From: 18_SqueezeNet

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_convolution_mean_relu_29(
    in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr
):
    xnumel = 1000
    RBLOCK: tl.constexpr = 2
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_indices = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = r_indices
    x0 = x_indices
    input_values = tl.load(in_ptr0 + (x0 + 1000 * r1), x_mask, other=0.0)
    broadcasted_values = tl.broadcast_to(input_values, [XBLOCK, RBLOCK])
    masked_values = tl.where(x_mask, broadcasted_values, 0)
    summed_values = tl.sum(masked_values, 1)[:, None]
    divisor = 169.0
    result = summed_values / divisor
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), result, x_mask)