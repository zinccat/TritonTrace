# From: 61_ConvTranspose3d_ReLU_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_group_norm_backward_3(
    input_grad_ptr, mean_ptr, inv_std_ptr, input_ptr, output_grad_ptr, weight_grad_ptr, 
    xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x1 = xindex // 16
    sum_grad_output = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    sum_weight_grad = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        input_grad = tl.load(input_grad_ptr + (x3 + 128 * r2), rmask & xmask, eviction_policy='evict_first', other=0.0)
        mean = tl.load(mean_ptr + (x3 + 128 * r2), rmask & xmask, eviction_policy='evict_first', other=0.0)
        inv_std = tl.load(inv_std_ptr + (x1 + 8 * r2), rmask & xmask, eviction_policy='evict_last', other=0.0)
        input_val = tl.load(input_ptr + (x1 + 8 * r2), rmask & xmask, eviction_policy='evict_last', other=0.0)

        mean_scaled = mean * inv_std
        grad_input_centered = input_grad - mean_scaled
        grad_input_scaled = grad_input_centered * inv_std

        grad_input_scaled_broadcast = tl.broadcast_to(grad_input_scaled, [XBLOCK, RBLOCK])
        sum_grad_output += grad_input_scaled_broadcast
        sum_grad_output = tl.where(rmask & xmask, sum_grad_output, sum_grad_output)

        weight_grad_broadcast = tl.broadcast_to(mean, [XBLOCK, RBLOCK])
        sum_weight_grad += weight_grad_broadcast
        sum_weight_grad = tl.where(rmask & xmask, sum_weight_grad, sum_weight_grad)

    output_grad_sum = tl.sum(sum_grad_output, 1)[:, None]
    weight_grad_sum = tl.sum(sum_weight_grad, 1)[:, None]

    tl.store(output_grad_ptr + (x3), output_grad_sum, xmask)
    tl.store(weight_grad_ptr + (x3), weight_grad_sum, xmask)