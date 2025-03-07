# From: 8_Conv3d_Divide_Max_GlobalAvgPool_BiasAdd_Sum

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_add_mean_sum_2(input_ptr0, input_ptr1, output_ptr0, kernel_size0, kernel_size1, num_elements_x, num_elements_r, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 16
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements_x
    r_indices = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = r_indices
    x0 = x_indices
    tmp0 = tl.load(input_ptr0 + (r1 + 16 * x0), x_mask, other=0.0)
    tmp4 = tl.load(input_ptr1 + (r1), None, eviction_policy='evict_last')
    tmp1 = (-1) + ((-1) * (kernel_size1 // 2) * (kernel_size1 // 2)) + 2 * (kernel_size1 // 2) + (kernel_size1 // 2) * (kernel_size1 // 2) * (kernel_size0 // 2) + ((-2) * (kernel_size0 // 2) * (kernel_size1 // 2)) + (kernel_size0 // 2)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 / tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(x_mask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tl.store(output_ptr0 + (x0), tmp9, x_mask)