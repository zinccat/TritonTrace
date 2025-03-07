# From: 11_ConvTranspose2d_BatchNorm_Tanh_MaxPool_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_native_batch_norm_backward_tanh_backward_7(input_grad_ptr, input_ptr, output_grad_ptr, output_ptr, input_num_elements, running_num_elements, XBLOCK: tl.constexpr):
    input_num_elements = 64
    running_num_elements = 6
    RBLOCK: tl.constexpr = 8
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < input_num_elements
    r_indices = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_indices < running_num_elements
    r1 = r_indices
    x0 = x_indices
    tmp0 = tl.load(input_grad_ptr + (x0 + 64 * r1), r_mask & x_mask, other=0.0)
    tmp5 = tl.load(input_ptr + (x0), x_mask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(r_mask & x_mask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(output_ptr + (x0), tmp6, x_mask)
    tl.store(output_grad_ptr + (x0), tmp4, x_mask)