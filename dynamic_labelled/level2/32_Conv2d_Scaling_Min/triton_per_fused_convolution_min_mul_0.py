# From: 32_Conv2d_Scaling_Min

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_convolution_min_mul_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, kernel_size0, kernel_size1, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 16
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < input_num_elements
    r_index = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = r_index
    x3 = (x_index % kernel_size0)
    x4 = x_index // kernel_size0
    x5 = x_index
    tmp0 = tl.load(in_ptr0 + (x3 + 4*r2 + 64*x4 + r2*kernel_size1*kernel_size1 + ((-64)*kernel_size1*x4) + ((-4)*kernel_size1*r2) + 16*x4*kernel_size1*kernel_size1), x_mask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 2.0
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(x_mask, tmp5, float("inf"))
    tmp8 = triton_helpers.min2(tmp7, 1)[:, None]
    tmp10 = tl.broadcast_to(r_index, tmp7.shape)
    tmp9_val, tmp9_idx = triton_helpers.min_with_index(tmp7, tmp10, 1)
    tmp9 = tmp9_idx[:, None]
    tl.store(out_ptr0 + (x5), tmp8, x_mask)
    tl.store(out_ptr1 + (x5), tmp9, x_mask)