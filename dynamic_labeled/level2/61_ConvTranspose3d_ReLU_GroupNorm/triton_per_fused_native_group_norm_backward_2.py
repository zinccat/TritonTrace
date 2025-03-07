# From: 61_ConvTranspose3d_ReLU_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_native_group_norm_backward_2(in_grad_input_ptr, in_grad_output_ptr, in_input_ptr, out_grad_input_ptr, out_grad_output_ptr, xnumel, rnumel, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_group_index = rindex
    x_block_index = xindex
    x_channel_index = xindex % 8
    grad_input = tl.load(in_grad_input_ptr + (r_group_index + 16 * x_block_index), xmask, other=0.0)
    grad_output = tl.load(in_grad_output_ptr + (r_group_index + 16 * x_channel_index), xmask, eviction_policy='evict_last', other=0.0)
    input_data = tl.load(in_input_ptr + (r_group_index + 16 * x_block_index), xmask, other=0.0)
    grad_input_times_grad_output = grad_input * grad_output
    broadcast_grad_input_times_grad_output = tl.broadcast_to(grad_input_times_grad_output, [XBLOCK, RBLOCK])
    masked_grad_input_times_grad_output = tl.where(xmask, broadcast_grad_input_times_grad_output, 0)
    sum_grad_input_times_grad_output = tl.sum(masked_grad_input_times_grad_output, 1)[:, None]
    input_data_times_grad_output = input_data * grad_output
    broadcast_input_data_times_grad_output = tl.broadcast_to(input_data_times_grad_output, [XBLOCK, RBLOCK])
    masked_input_data_times_grad_output = tl.where(xmask, broadcast_input_data_times_grad_output, 0)
    sum_input_data_times_grad_output = tl.sum(masked_input_data_times_grad_output, 1)[:, None]
    tl.store(out_grad_input_ptr + (x_block_index), sum_grad_input_times_grad_output, xmask)
    tl.store(out_grad_output_ptr + (x_block_index), sum_input_data_times_grad_output, xmask)