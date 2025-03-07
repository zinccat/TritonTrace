# From: 34_ConvTranspose3d_LayerNorm_GELU_Scaling

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_gelu_gelu_backward_mul_native_layer_norm_native_layer_norm_backward_0(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr
):
    RBLOCK: tl.constexpr = 64
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_index = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = r_index
    x0 = x_index

    input0 = tl.load(in_ptr0 + (r1 + 64 * x0), None)
    input1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    input3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    input5 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    input7 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    input9 = tl.load(in_ptr5 + (r1 + 64 * x0), None)

    diff = input0 - input1
    scaled_diff = diff * input3
    scaled_diff_input5 = scaled_diff * input5
    sum_scaled_diff_input5_input7 = scaled_diff_input5 + input7

    constant1 = 1.0
    scaled_input9 = input9 * constant1
    constant2 = 0.7071067811865476
    scaled_sum = sum_scaled_diff_input5_input7 * constant2
    erf_result = tl.extra.cuda.libdevice.erf(scaled_sum)
    erf_result_plus_one = erf_result + constant1
    constant3 = 0.5
    half_erf_result_plus_one = erf_result_plus_one * constant3

    squared_sum = sum_scaled_diff_input5_input7 * sum_scaled_diff_input5_input7
    constant4 = -0.5
    exp_argument = squared_sum * constant4
    exp_result = tl.math.exp(exp_argument)
    constant5 = 0.3989422804014327
    scaled_exp_result = exp_result * constant5
    scaled_sum_exp_result = sum_scaled_diff_input5_input7 * scaled_exp_result
    gelu_result = half_erf_result_plus_one + scaled_sum_exp_result

    scaled_gelu_result = scaled_input9 * gelu_result
    scaled_gelu_result_input5 = scaled_gelu_result * input5
    broadcast_scaled_gelu_result_input5 = tl.broadcast_to(scaled_gelu_result_input5, [XBLOCK, RBLOCK])
    sum_broadcast_scaled_gelu_result_input5 = tl.sum(broadcast_scaled_gelu_result_input5, 1)[:, None]

    scaled_gelu_result_scaled_diff = scaled_gelu_result_input5 * scaled_diff
    broadcast_scaled_gelu_result_scaled_diff = tl.broadcast_to(scaled_gelu_result_scaled_diff, [XBLOCK, RBLOCK])
    sum_broadcast_scaled_gelu_result_scaled_diff = tl.sum(broadcast_scaled_gelu_result_scaled_diff, 1)[:, None]

    constant6 = 64.0
    scaled_gelu_result_constant6 = scaled_gelu_result_input5 * constant6
    diff_sum_broadcast_scaled_gelu_result_constant6 = scaled_gelu_result_constant6 - sum_broadcast_scaled_gelu_result_input5
    product_scaled_diff_sum_broadcast_scaled_gelu_result_scaled_diff = scaled_diff * sum_broadcast_scaled_gelu_result_scaled_diff
    final_diff = diff_sum_broadcast_scaled_gelu_result_constant6 - product_scaled_diff_sum_broadcast_scaled_gelu_result_scaled_diff

    constant7 = 0.015625
    scaled_input3_constant7 = input3 * constant7
    final_result = scaled_input3_constant7 * final_diff

    tl.store(out_ptr0 + (r1 + 64 * x0), sum_scaled_diff_input5_input7, None)
    tl.store(in_out_ptr0 + (r1 + 64 * x0), final_result, None)