# From: 23_Conv3d_GroupNorm_Mean

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_group_norm_backward_1red_fused_native_group_norm_backward_1(
    input_grad_ptr, input_ptr, mean_ptr, inv_std_ptr, output_grad_ptr, output_ptr, 
    xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):

    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x1 = xindex // 2
    sum_grad_xr = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    sum_input_xr = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        grad_input = tl.load(input_grad_ptr + (x3 + 16 * r2), rmask & xmask, eviction_policy='evict_first', other=0.0)
        input_data = tl.load(input_ptr + (x3 + 16 * r2), rmask & xmask, eviction_policy='evict_first', other=0.0)
        mean = tl.load(mean_ptr + (x1 + 8 * r2), rmask & xmask, eviction_policy='evict_last', other=0.0)
        inv_std = tl.load(inv_std_ptr + (x1 + 8 * r2), rmask & xmask, eviction_policy='evict_last', other=0.0)

        input_mean = input_data * mean
        grad_input_centered = grad_input - input_mean
        grad_output = grad_input_centered * inv_std
        grad_output_broadcast = tl.broadcast_to(grad_output, [XBLOCK, RBLOCK])

        sum_grad_xr_temp = sum_grad_xr + grad_output_broadcast
        sum_grad_xr = tl.where(rmask & xmask, sum_grad_xr_temp, sum_grad_xr)

        input_data_broadcast = tl.broadcast_to(input_data, [XBLOCK, RBLOCK])
        sum_input_xr_temp = sum_input_xr + input_data_broadcast
        sum_input_xr = tl.where(rmask & xmask, sum_input_xr_temp, sum_input_xr)

    sum_grad_x = tl.sum(sum_grad_xr, 1)[:, None]
    sum_input_x = tl.sum(sum_input_xr, 1)[:, None]

    tl.store(output_grad_ptr + (x3), sum_grad_x, xmask)
    tl.store(output_ptr + (x3), sum_input_x, xmask)