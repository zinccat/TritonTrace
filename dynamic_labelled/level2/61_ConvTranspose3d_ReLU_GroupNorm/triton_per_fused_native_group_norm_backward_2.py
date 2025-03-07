# From: 61_ConvTranspose3d_ReLU_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_native_group_norm_backward_2(
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
    x_mod_index = xindex % 8
    grad_input = tl.load(input_grad_ptr + (r_block_index + 16 * x_block_index), xmask, other=0.0)
    input_data = tl.load(input_ptr + (r_block_index + 16 * x_mod_index), xmask, eviction_policy='evict_last', other=0.0)
    running_var = tl.load(running_var_ptr + (r_block_index + 16 * x_block_index), xmask, other=0.0)
    
    grad_input_times_input = grad_input * input_data
    broadcast_grad_input_times_input = tl.broadcast_to(grad_input_times_input, [XBLOCK, RBLOCK])
    masked_grad_input_times_input = tl.where(xmask, broadcast_grad_input_times_input, 0)
    sum_grad_input_times_input = tl.sum(masked_grad_input_times_input, 1)[:, None]
    
    running_var_times_input = running_var * input_data
    broadcast_running_var_times_input = tl.broadcast_to(running_var_times_input, [XBLOCK, RBLOCK])
    masked_running_var_times_input = tl.where(xmask, broadcast_running_var_times_input, 0)
    sum_running_var_times_input = tl.sum(masked_running_var_times_input, 1)[:, None]
    
    tl.store(output_grad_ptr + (x_block_index), sum_grad_input_times_input, xmask)
    tl.store(output_ptr + (x_block_index), sum_running_var_times_input, xmask)