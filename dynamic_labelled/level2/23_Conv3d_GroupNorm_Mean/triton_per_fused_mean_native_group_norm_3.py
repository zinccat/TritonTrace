# From: 23_Conv3d_GroupNorm_Mean

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_mean_native_group_norm_3(in_out_ptr0, in_ptr0, kernel_size0, kernel_size1, x_num_elements, r_num_elements, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 4
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < x_num_elements
    r_indices = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = r_indices
    x0 = x_indices
    tmp0 = tl.load(in_ptr0 + (r1 + 4 * x0), x_mask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(x_mask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = (-128) + ((-32) * kernel_size1 * kernel_size1) + 64 * kernel_size0 + 128 * kernel_size1 + ((-64) * kernel_size0 * kernel_size1) + 16 * kernel_size0 * kernel_size1 * kernel_size1
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp4 / tmp6
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp7, x_mask)