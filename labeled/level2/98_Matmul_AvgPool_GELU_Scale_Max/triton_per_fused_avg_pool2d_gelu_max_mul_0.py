# From: 98_Matmul_AvgPool_GELU_Scale_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_per_fused_avg_pool2d_gelu_max_mul_0(input_ptr, output_ptr_avg, output_ptr_max, output_ptr_index, num_elements, reduction_num_elements, XBLOCK: tl.constexpr):
    num_elements = 128
    RBLOCK: tl.constexpr = 64
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements
    r_indices = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = r_indices
    x0 = x_indices

    tmp0 = tl.load(input_ptr + ((4 * r1) + (256 * x0)), x_mask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(input_ptr + (1 + (4 * r1) + (256 * x0)), x_mask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(input_ptr + (2 + (4 * r1) + (256 * x0)), x_mask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(input_ptr + (3 + (4 * r1) + (256 * x0)), x_mask, eviction_policy='evict_last', other=0.0)

    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4

    tmp7 = 0.25
    tmp8 = tmp6 * tmp7

    tmp9 = 0.5
    tmp10 = tmp8 * tmp9

    tmp11 = 0.7071067811865476
    tmp12 = tmp8 * tmp11

    tmp13 = tl.extra.cuda.libdevice.erf(tmp12)

    tmp14 = 1.0
    tmp15 = tmp13 + tmp14

    tmp16 = tmp10 * tmp15

    tmp17 = 2.0
    tmp18 = tmp16 * tmp17

    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])

    tmp21 = tl.where(x_mask, tmp19, float("-inf"))

    tmp22 = triton_helpers.max2(tmp21, 1)[:, None]

    tmp24 = tl.broadcast_to(r_indices, tmp21.shape)
    _, tmp23_tmp = triton_helpers.max_with_index(tmp21, tmp24, 1)

    tmp23 = tmp23_tmp[:, None]

    tl.store(output_ptr_avg + (r1 + (64 * x0)), tmp8, x_mask)
    tl.store(output_ptr_max + (x0), tmp22, x_mask)
    tl.store(output_ptr_index + (x0), tmp23, x_mask)