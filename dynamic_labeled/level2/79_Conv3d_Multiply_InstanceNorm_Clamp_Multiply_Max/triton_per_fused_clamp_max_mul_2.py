# From: 79_Conv3d_Multiply_InstanceNorm_Clamp_Multiply_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_clamp_max_mul_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, kernel_size0, kernel_size1, kernel_size2, xnumel, rnumel, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 16
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_index = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = r_index
    x3 = (x_index % kernel_size0)
    x4 = x_index // kernel_size0
    x5 = x_index
    tmp0 = tl.load(in_ptr0 + (x3 + ((-128)*x4) + ((-8)*r2) + ((-32)*x4*kernel_size2*kernel_size2) + ((-2)*r2*kernel_size2*kernel_size2) + 4*kernel_size1*r2 + 8*kernel_size2*r2 + 64*kernel_size1*x4 + 128*kernel_size2*x4 + kernel_size1*r2*kernel_size2*kernel_size2 + ((-64)*kernel_size1*kernel_size2*x4) + ((-4)*kernel_size1*kernel_size2*r2) + 16*kernel_size1*x4*kernel_size2*kernel_size2), x_mask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (r2 + 16*x4), x_mask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr3 + (r2 + 16*x4), x_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = tmp4 * tmp5
    tmp7 = -1.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = 1.0
    tmp10 = triton_helpers.minimum(tmp8, tmp9)
    tmp11 = tmp10 * tmp1
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
    tmp14 = tl.where(x_mask, tmp12, float("-inf"))
    tmp15 = triton_helpers.max2(tmp14, 1)[:, None]
    tmp17 = tl.broadcast_to(r_index, tmp14.shape)
    tmp16_val, tmp16_idx = triton_helpers.max_with_index(tmp14, tmp17, 1)
    tmp16 = tmp16_idx[:, None]
    tl.store(out_ptr0 + (x5), tmp15, x_mask)
    tl.store(out_ptr1 + (x5), tmp16, x_mask)