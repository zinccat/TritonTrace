# From: 41_Gemm_BatchNorm_GELU_GroupNorm_Mean_ReLU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_group_norm_backward_2(
    input_grad_ptr, input_mean_ptr, input_var_ptr, input_weight_ptr, input_bias_ptr,
    output_grad_ptr, output_mean_ptr, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x1 = xindex // 128
    sum_grad_weight = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    sum_weight = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        grad_mask = tl.load(input_grad_ptr + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
        grad_input = tl.load(input_mean_ptr + (r2), rmask, eviction_policy='evict_last', other=0.0)
        input_var = tl.load(input_var_ptr + (x3 + 1024 * r2), rmask & xmask, eviction_policy='evict_first', other=0.0)
        weight = tl.load(input_weight_ptr + (x1 + 8 * r2), rmask & xmask, eviction_policy='evict_last', other=0.0)
        bias = tl.load(input_bias_ptr + (x1 + 8 * r2), rmask & xmask, eviction_policy='evict_last', other=0.0)

        grad_input_selected = tl.where(grad_mask, 0.0, grad_input)
        scale_factor = 0.0009765625
        scaled_grad_input = grad_input_selected * scale_factor

        half = 0.5
        scaled_input_var = input_var * half
        sqrt_2_over_sqrt_pi = 0.7071067811865476
        erf_input = input_var * sqrt_2_over_sqrt_pi
        erf_result = tl.extra.cuda.libdevice.erf(erf_input)
        erf_result_scaled = scaled_input_var * (erf_result + 1.0)

        grad_weight = scaled_grad_input * erf_result_scaled
        grad_weight_scaled = grad_weight * weight
        grad_input_weight = grad_weight - grad_weight_scaled
        grad_input_weight_scaled = grad_input_weight * bias

        grad_input_weight_broadcast = tl.broadcast_to(grad_input_weight_scaled, [XBLOCK, RBLOCK])
        sum_grad_weight += grad_input_weight_broadcast
        sum_grad_weight = tl.where(rmask & xmask, sum_grad_weight, sum_grad_weight)

        scaled_grad_input_broadcast = tl.broadcast_to(scaled_grad_input, [XBLOCK, RBLOCK])
        sum_weight += scaled_grad_input_broadcast
        sum_weight = tl.where(rmask & xmask, sum_weight, sum_weight)

    sum_grad_weight_final = tl.sum(sum_grad_weight, 1)[:, None]
    sum_weight_final = tl.sum(sum_weight, 1)[:, None]

    tl.store(output_grad_ptr + (x3), sum_grad_weight_final, xmask)
    tl.store(output_mean_ptr + (x3), sum_weight_final, xmask)