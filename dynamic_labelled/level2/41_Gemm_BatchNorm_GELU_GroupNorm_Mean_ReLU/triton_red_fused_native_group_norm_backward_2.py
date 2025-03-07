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
    sum_grad = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        grad_mask = tl.load(input_grad_ptr + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
        grad = tl.load(input_mean_ptr + (r2), rmask, eviction_policy='evict_last', other=0.0)
        input_var = tl.load(input_var_ptr + (x3 + 1024 * r2), rmask & xmask, eviction_policy='evict_first', other=0.0)
        weight = tl.load(input_weight_ptr + (x1 + 8 * r2), rmask & xmask, eviction_policy='evict_last', other=0.0)
        bias = tl.load(input_bias_ptr + (x1 + 8 * r2), rmask & xmask, eviction_policy='evict_last', other=0.0)

        grad_without_mask = tl.where(grad_mask, 0.0, grad)
        scale_factor = 0.0009765625
        scaled_grad = grad_without_mask * scale_factor
        half = 0.5
        scaled_var = input_var * half
        sqrt_2_over_sqrt_pi = 0.7071067811865476
        scaled_var_sqrt_2_over_sqrt_pi = input_var * sqrt_2_over_sqrt_pi
        erf_result = tl.extra.cuda.libdevice.erf(scaled_var_sqrt_2_over_sqrt_pi)
        one = 1.0
        erf_plus_one = erf_result + one
        scaled_var_erf_plus_one = scaled_var * erf_plus_one
        scaled_grad_scaled_var_erf_plus_one = scaled_grad * scaled_var_erf_plus_one
        scaled_grad_weight = scaled_grad * weight
        grad_weight_diff = scaled_grad_scaled_var_erf_plus_one - scaled_grad_weight
        grad_weight_diff_scaled_bias = grad_weight_diff * bias
        broadcast_grad_weight_diff_scaled_bias = tl.broadcast_to(grad_weight_diff_scaled_bias, [XBLOCK, RBLOCK])
        sum_grad_weight += broadcast_grad_weight_diff_scaled_bias
        broadcast_scaled_grad = tl.broadcast_to(scaled_grad, [XBLOCK, RBLOCK])
        sum_grad += broadcast_scaled_grad

        sum_grad_weight = tl.where(rmask & xmask, sum_grad_weight, sum_grad_weight)
        sum_grad = tl.where(rmask & xmask, sum_grad, sum_grad)

    output_grad_sum = tl.sum(sum_grad_weight, 1)[:, None]
    output_mean_sum = tl.sum(sum_grad, 1)[:, None]
    tl.store(output_grad_ptr + (x3), output_grad_sum, xmask)
    tl.store(output_mean_ptr + (x3), output_mean_sum, xmask)