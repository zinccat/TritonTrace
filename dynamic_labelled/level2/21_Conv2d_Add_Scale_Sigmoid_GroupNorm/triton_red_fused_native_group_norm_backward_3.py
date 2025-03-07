# From: 21_Conv2d_Add_Scale_Sigmoid_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_group_norm_backward_3(
    input_grad_ptr, input_ptr, scale_ptr, bias_ptr, output_grad_ptr, output_ptr, 
    xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x1 = xindex // 2
    sum_grad_input = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    sum_scale = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        grad_input = tl.load(input_grad_ptr + (x3 + 16 * r2), rmask & xmask, eviction_policy='evict_first', other=0.0)
        input_value = tl.load(input_ptr + (x3 + 16 * r2), rmask & xmask, eviction_policy='evict_first', other=0.0)
        scale_value = tl.load(scale_ptr + (x1 + 8 * r2), rmask & xmask, eviction_policy='evict_last', other=0.0)
        bias_value = tl.load(bias_ptr + (x1 + 8 * r2), rmask & xmask, eviction_policy='evict_last', other=0.0)

        scaled_input = input_value * scale_value
        grad_input_diff = grad_input - scaled_input
        grad_input_scaled = grad_input_diff * bias_value
        broadcast_grad_input_scaled = tl.broadcast_to(grad_input_scaled, [XBLOCK, RBLOCK])

        sum_grad_input += broadcast_grad_input_scaled
        sum_grad_input = tl.where(rmask & xmask, sum_grad_input, sum_grad_input)

        broadcast_scale = tl.broadcast_to(scale_value, [XBLOCK, RBLOCK])
        sum_scale += broadcast_scale
        sum_scale = tl.where(rmask & xmask, sum_scale, sum_scale)

    total_grad_input = tl.sum(sum_grad_input, 1)[:, None]
    total_scale = tl.sum(sum_scale, 1)[:, None]

    tl.store(output_grad_ptr + (x3), total_grad_input, xmask)
    tl.store(output_ptr + (x3), total_scale, xmask)