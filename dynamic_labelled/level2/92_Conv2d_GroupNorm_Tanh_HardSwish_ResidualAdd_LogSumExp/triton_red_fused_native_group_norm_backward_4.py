# From: 92_Conv2d_GroupNorm_Tanh_HardSwish_ResidualAdd_LogSumExp

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_group_norm_backward_4red_fused_native_group_norm_backward_4(
    input_grad_ptr, input_ptr, mean_ptr, variance_ptr, output_grad_ptr, output_mean_ptr, 
    xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):

    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x1 = xindex // 2

    temp_sum_grad = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    temp_sum_input = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex

        grad_input = tl.load(input_grad_ptr + (x3 + 16 * r2), rmask & xmask, eviction_policy='evict_first', other=0.0)
        input_data = tl.load(input_ptr + (x3 + 16 * r2), rmask & xmask, eviction_policy='evict_first', other=0.0)
        mean = tl.load(mean_ptr + (x1 + 8 * r2), rmask & xmask, eviction_policy='evict_last', other=0.0)
        inv_std = tl.load(variance_ptr + (x1 + 8 * r2), rmask & xmask, eviction_policy='evict_last', other=0.0)

        normalized_input = input_data * mean
        grad_normalized_input = grad_input - normalized_input
        grad_input_scaled = grad_normalized_input * inv_std

        broadcast_grad_input_scaled = tl.broadcast_to(grad_input_scaled, [XBLOCK, RBLOCK])
        temp_sum_grad += broadcast_grad_input_scaled
        temp_sum_grad = tl.where(rmask & xmask, temp_sum_grad, temp_sum_grad)

        broadcast_input_data = tl.broadcast_to(input_data, [XBLOCK, RBLOCK])
        temp_sum_input += broadcast_input_data
        temp_sum_input = tl.where(rmask & xmask, temp_sum_input, temp_sum_input)

    sum_grad = tl.sum(temp_sum_grad, 1)[:, None]
    sum_input = tl.sum(temp_sum_input, 1)[:, None]

    tl.store(output_grad_ptr + (x3), sum_grad, xmask)
    tl.store(output_mean_ptr + (x3), sum_input, xmask)