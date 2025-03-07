# From: 49_ConvTranspose3d_Softmax_Sigmoid

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__softmax_1(in_ptr0, out_ptr0, out_ptr1, kernel_size0, kernel_size1, xnumel, rnumel, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 64
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_indices = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = r_indices
    x0 = (x_indices % kernel_size0)
    x1 = x_indices // kernel_size0
    x3 = x_indices
    input_values = tl.load(in_ptr0 + (x0 + 8192 * kernel_size1 * r2 + 524288 * kernel_size1 * x1), None, eviction_policy='evict_last')
    broadcasted_values = tl.broadcast_to(input_values, [XBLOCK, RBLOCK])
    max_values = triton_helpers.max2(broadcasted_values, 1)[:, None]
    shifted_values = input_values - max_values
    exp_values = tl.math.exp(shifted_values)
    broadcasted_exp_values = tl.broadcast_to(exp_values, [XBLOCK, RBLOCK])
    sum_exp_values = tl.sum(broadcasted_exp_values, 1)[:, None]
    tl.store(out_ptr0 + (x3), max_values, None)
    tl.store(out_ptr1 + (x3), sum_exp_values, None)