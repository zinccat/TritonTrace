# From: 89_ConvTranspose3d_MaxPool_Softmax_Subtract_Swish_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_per_fused_max_mul_sigmoid_sub_3(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 16
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_indices = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = r_indices
    x0 = x_indices % 16384
    x1 = (x_indices // 16384)
    x3 = x_indices
    input0_value = tl.load(in_ptr0 + (x0 + (16384 * r2) + (262144 * x1)), None)
    input1_value = tl.load(in_ptr1 + (r2), None, eviction_policy='evict_last')
    subtracted_value = input0_value - input1_value
    sigmoid_value = tl.sigmoid(subtracted_value)
    multiplied_value = sigmoid_value * subtracted_value
    broadcasted_value = tl.broadcast_to(multiplied_value, [XBLOCK, RBLOCK])
    max_value = triton_helpers.max2(broadcasted_value, 1)[:, None]
    broadcasted_r_indices = tl.broadcast_to(r_indices, broadcasted_value.shape)
    _, max_index = triton_helpers.max_with_index(broadcasted_value, broadcasted_r_indices, 1)
    max_index = max_index[:, None]
    tl.store(out_ptr0 + (x3), max_value, None)
    tl.store(out_ptr1 + (x3), max_index, None)