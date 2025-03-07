# From: 6_Conv3d_Softmax_MaxPool_MaxPool

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__softmax_convolution_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ks0, ks1, ks2, xnumel, rnumel, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    row_index = rindex
    col_index_mod_ks0 = xindex % ks0
    col_index_div_ks0 = xindex // ks0
    linear_index = xindex

    input_offset = (
        col_index_mod_ks0 +
        ((-128) * col_index_div_ks0) +
        ((-8) * row_index) +
        ((-32) * col_index_div_ks0 * ks2 * ks2) +
        ((-2) * row_index * ks2 * ks2) +
        4 * ks1 * row_index +
        8 * ks2 * row_index +
        64 * ks1 * col_index_div_ks0 +
        128 * ks2 * col_index_div_ks0 +
        ks1 * row_index * ks2 * ks2 +
        ((-64) * ks1 * ks2 * col_index_div_ks0) +
        ((-4) * ks1 * ks2 * row_index) +
        16 * ks1 * col_index_div_ks0 * ks2 * ks2
    )

    tmp0 = tl.load(in_ptr0 + input_offset, xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + (row_index), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(xmask, tmp3, float("-inf"))
    max_values = triton_helpers.max2(tmp5, 1)[:, None]
    tmp7 = tmp2 - max_values
    exp_values = tl.math.exp(tmp7)
    tmp9 = tl.broadcast_to(exp_values, [XBLOCK, RBLOCK])
    masked_exp_values = tl.where(xmask, tmp9, 0)
    sum_exp_values = tl.sum(masked_exp_values, 1)[:, None]

    tl.store(out_ptr0 + (linear_index), max_values, xmask)
    tl.store(out_ptr1 + (linear_index), sum_exp_values, xmask)