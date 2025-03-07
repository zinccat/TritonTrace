# From: 36_ConvTranspose2d_Min_Sum_GELU_Add

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_per_fused_convolution_min_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 16
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_indices = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = r_indices
    x0 = x_indices % 4096
    x1 = (x_indices // 4096)
    x3 = x_indices
    input0 = tl.load(in_ptr0 + (x0 + (4096 * r2) + (65536 * x1)), None)
    input1 = tl.load(in_ptr1 + (r2), None, eviction_policy='evict_last')
    sum_result = input0 + input1
    broadcast_sum = tl.broadcast_to(sum_result, [XBLOCK, RBLOCK])
    min_values = triton_helpers.min2(broadcast_sum, 1)[:, None]
    index_broadcast = tl.broadcast_to(r_indices, broadcast_sum.shape)
    _, min_indices = triton_helpers.min_with_index(broadcast_sum, index_broadcast, 1)
    min_indices = min_indices[:, None]
    tl.store(out_ptr0 + (x3), min_values, None)
    tl.store(out_ptr1 + (x3), min_indices, None)