# From: 89_ConvTranspose3d_MaxPool_Softmax_Subtract_Swish_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__softmax_1(in_ptr0, out_ptr0, out_ptr1, ks0, ks1, ks2, xnumel, rnumel, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    x3 = xindex
    input_slice = tl.load(in_ptr0 + (x0 + ks1 * r2 * ks2 * ks2 + 16 * ks1 * x1 * ks2 * ks2), xmask, eviction_policy='evict_last', other=0.0)
    broadcasted_input = tl.broadcast_to(input_slice, [XBLOCK, RBLOCK])
    masked_input = tl.where(xmask, broadcasted_input, float("-inf"))
    max_values = triton_helpers.max2(masked_input, 1)[:, None]
    shifted_input = input_slice - max_values
    exp_values = tl.math.exp(shifted_input)
    broadcasted_exp = tl.broadcast_to(exp_values, [XBLOCK, RBLOCK])
    masked_exp = tl.where(xmask, broadcasted_exp, 0)
    sum_exp = tl.sum(masked_exp, 1)[:, None]
    tl.store(out_ptr0 + (x3), max_values, xmask)
    tl.store(out_ptr1 + (x3), sum_exp, xmask)