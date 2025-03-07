# From: 30_Gemm_GroupNorm_Hardtanh

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_group_norm_backward_1(
    input_grad_ptr, input_ptr, mean_ptr, variance_ptr, inv_std_ptr, 
    output_grad_ptr, output_ptr, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x1 = xindex // 64
    sum_grad_output = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    sum_grad_input = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex

        grad_output = tl.load(input_grad_ptr + (x3 + 512 * r2), rmask & xmask, eviction_policy='evict_first', other=0.0)
        input = tl.load(input_ptr + (x3 + 512 * r2), rmask & xmask, eviction_policy='evict_first', other=0.0)
        variance = tl.load(variance_ptr + (x3 + 512 * r2), rmask & xmask, eviction_policy='evict_first', other=0.0)
        mean = tl.load(mean_ptr + (x1 + 8 * r2), rmask & xmask, eviction_policy='evict_last', other=0.0)
        inv_std = tl.load(inv_std_ptr + (x1 + 8 * r2), rmask & xmask, eviction_policy='evict_last', other=0.0)

        lower_bound = -2.0
        upper_bound = 2.0
        is_out_of_bounds = (input <= lower_bound) | (input >= upper_bound)
        clamped_input = tl.where(is_out_of_bounds, 0.0, input)

        grad_input = clamped_input * variance
        grad_mean = grad_input * mean
        grad_variance = grad_input - grad_mean
        scaled_grad_variance = grad_variance * inv_std

        broadcast_scaled_grad_variance = tl.broadcast_to(scaled_grad_variance, [XBLOCK, RBLOCK])
        sum_grad_output += broadcast_scaled_grad_variance
        sum_grad_output = tl.where(rmask & xmask, sum_grad_output, sum_grad_output)

        broadcast_clamped_input = tl.broadcast_to(clamped_input, [XBLOCK, RBLOCK])
        sum_grad_input += broadcast_clamped_input
        sum_grad_input = tl.where(rmask & xmask, sum_grad_input, sum_grad_input)

    total_grad_output = tl.sum(sum_grad_output, 1)[:, None]
    total_grad_input = tl.sum(sum_grad_input, 1)[:, None]

    tl.store(output_grad_ptr + (x3), total_grad_output, xmask)
    tl.store(output_ptr + (x3), total_grad_input, xmask)