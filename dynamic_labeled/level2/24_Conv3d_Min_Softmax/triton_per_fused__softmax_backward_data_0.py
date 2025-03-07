# From: 24_Conv3d_Min_Softmax

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__softmax_backward_data_0(input_grad_ptr, output_ptr, output_grad_ptr, kernel_size_dim0, kernel_size_dim1, num_elements_x, num_elements_r, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 16
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < num_elements_x
    r_index = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = r_index
    x3 = (x_index % kernel_size_dim0)
    x4 = x_index // kernel_size_dim0
    x5 = x_index
    tmp0 = tl.load(input_grad_ptr + (x3 + 4*r2 + 64*x4 + r2*kernel_size_dim1*kernel_size_dim1 + ((-64)*kernel_size_dim1*x4) + ((-4)*kernel_size_dim1*r2) + 16*x4*kernel_size_dim1*kernel_size_dim1), x_mask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(output_ptr + (x3 + 4*r2 + 64*x4 + r2*kernel_size_dim1*kernel_size_dim1 + ((-64)*kernel_size_dim1*x4) + ((-4)*kernel_size_dim1*r2) + 16*x4*kernel_size_dim1*kernel_size_dim1), x_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(x_mask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tl.store(output_grad_ptr + (x5), tmp6, x_mask)