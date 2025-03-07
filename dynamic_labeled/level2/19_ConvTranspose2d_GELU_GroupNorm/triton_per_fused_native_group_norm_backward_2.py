# From: 19_ConvTranspose2d_GELU_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_native_group_norm_backward_2(input_grad_ptr, mean_ptr, inv_std_ptr, output_grad_ptr, output_mean_ptr, xnumel, rnumel, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    row_indices = rindex
    col_indices = xindex
    col_mod8 = xindex % 8
    grad_input = tl.load(input_grad_ptr + (row_indices + 8 * col_indices), xmask, other=0.0)
    mean_values = tl.load(mean_ptr + (row_indices + 8 * col_mod8), xmask, eviction_policy='evict_last', other=0.0)
    inv_std_values = tl.load(inv_std_ptr + (row_indices + 8 * col_indices), xmask, other=0.0)
    
    grad_input_mean = grad_input * mean_values
    broadcast_grad_input_mean = tl.broadcast_to(grad_input_mean, [XBLOCK, RBLOCK])
    masked_grad_input_mean = tl.where(xmask, broadcast_grad_input_mean, 0)
    sum_grad_input_mean = tl.sum(masked_grad_input_mean, 1)[:, None]
    
    inv_std_mean = inv_std_values * mean_values
    broadcast_inv_std_mean = tl.broadcast_to(inv_std_mean, [XBLOCK, RBLOCK])
    masked_inv_std_mean = tl.where(xmask, broadcast_inv_std_mean, 0)
    sum_inv_std_mean = tl.sum(masked_inv_std_mean, 1)[:, None]
    
    tl.store(output_grad_ptr + (col_indices), sum_grad_input_mean, xmask)
    tl.store(output_mean_ptr + (col_indices), sum_inv_std_mean, xmask)