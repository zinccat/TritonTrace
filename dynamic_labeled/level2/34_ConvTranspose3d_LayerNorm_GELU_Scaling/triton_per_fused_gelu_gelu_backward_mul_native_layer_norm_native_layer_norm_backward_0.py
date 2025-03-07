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
    scaled_diff_layer_norm = scaled_diff * input5
    layer_norm_output = scaled_diff_layer_norm + input7

    one = 1.0
    sqrt_two = 0.7071067811865476
    erf_input = layer_norm_output * sqrt_two
    erf_output = tl.extra.cuda.libdevice.erf(erf_input)
    erf_plus_one = erf_output + one
    half = 0.5
    gelu_approx = erf_plus_one * half

    squared_output = layer_norm_output * layer_norm_output
    neg_half = -0.5
    exp_input = squared_output * neg_half
    exp_output = tl.math.exp(exp_input)
    sqrt_two_pi = 0.3989422804014327
    gaussian = exp_output * sqrt_two_pi
    gaussian_scaled = layer_norm_output * gaussian
    gelu_exact = gelu_approx + gaussian_scaled

    gelu_scaled = input9 * one
    gelu_result = gelu_scaled * gelu_exact
    gelu_scaled_layer_norm = gelu_result * input5

    broadcast_gelu_scaled_layer_norm = tl.broadcast_to(gelu_scaled_layer_norm, [XBLOCK, RBLOCK])
    sum_gelu_scaled_layer_norm = tl.sum(broadcast_gelu_scaled_layer_norm, 1)[:, None]

    gelu_scaled_diff = gelu_scaled_layer_norm * scaled_diff
    broadcast_gelu_scaled_diff = tl.broadcast_to(gelu_scaled_diff, [XBLOCK, RBLOCK])
    sum_gelu_scaled_diff = tl.sum(broadcast_gelu_scaled_diff, 1)[:, None]

    sixty_four = 64.0
    gelu_scaled_sixty_four = gelu_scaled_layer_norm * sixty_four
    gelu_scaled_sixty_four_minus_sum = gelu_scaled_sixty_four - sum_gelu_scaled_layer_norm
    gelu_scaled_diff_times_sum = scaled_diff * sum_gelu_scaled_diff
    final_result = gelu_scaled_sixty_four_minus_sum - gelu_scaled_diff_times_sum

    scaling_factor = 0.015625
    input3_scaled = input3 * scaling_factor
    final_scaled_result = input3_scaled * final_result

    tl.store(out_ptr0 + (r1 + 64 * x0), layer_norm_output, None)
    tl.store(in_out_ptr0 + (r1 + 64 * x0), final_scaled_result, None)