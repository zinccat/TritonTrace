# From: 11_ConvTranspose2d_BatchNorm_Tanh_MaxPool_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_native_group_norm_backward_1(
    input_grad_ptr, input_ptr, running_var_ptr, output_grad_ptr, output_ptr, 
    xnumel, rnumel, XBLOCK: tl.constexpr
):
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_block_index = rindex
    x_block_index = xindex
    x_channel_index = xindex % 4

    input_grad = tl.load(input_grad_ptr + (r_block_index + 16 * x_block_index), xmask, other=0.0)
    input = tl.load(input_ptr + (r_block_index + 16 * x_channel_index), xmask, eviction_policy='evict_last', other=0.0)
    running_var = tl.load(running_var_ptr + (r_block_index + 16 * x_block_index), xmask, other=0.0)

    grad_input_product = input_grad * input
    broadcast_grad_input_product = tl.broadcast_to(grad_input_product, [XBLOCK, RBLOCK])
    masked_grad_input_product = tl.where(xmask, broadcast_grad_input_product, 0)
    sum_grad_input_product = tl.sum(masked_grad_input_product, 1)[:, None]

    grad_running_var_product = running_var * input
    broadcast_grad_running_var_product = tl.broadcast_to(grad_running_var_product, [XBLOCK, RBLOCK])
    masked_grad_running_var_product = tl.where(xmask, broadcast_grad_running_var_product, 0)
    sum_grad_running_var_product = tl.sum(masked_grad_running_var_product, 1)[:, None]

    tl.store(output_grad_ptr + (x_block_index), sum_grad_input_product, xmask)
    tl.store(output_ptr + (x_block_index), sum_grad_running_var_product, xmask)