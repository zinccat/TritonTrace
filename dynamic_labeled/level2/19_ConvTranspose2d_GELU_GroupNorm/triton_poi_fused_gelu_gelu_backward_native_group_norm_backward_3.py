# From: 19_ConvTranspose2d_GELU_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_gelu_gelu_backward_native_group_norm_backward_3(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel

    # Load inputs
    input0 = tl.load(in_ptr0 + (xindex), xmask)
    input1 = tl.load(in_ptr1 + (xindex // 34848), xmask, eviction_policy='evict_last')
    input2 = tl.load(in_ptr2 + ((xindex // 4356) % 64), xmask, eviction_policy='evict_last')
    input3 = tl.load(in_ptr3 + (xindex), xmask)
    input4 = tl.load(in_ptr4 + (xindex // 34848), xmask, eviction_policy='evict_last')
    input5 = tl.load(in_ptr5 + (xindex // 34848), xmask, eviction_policy='evict_last')
    input6 = tl.load(in_ptr6 + (xindex // 34848), xmask, eviction_policy='evict_last')

    # Intermediate calculations
    product1 = input1 * input2
    product2 = input0 * product1
    half_input3 = input3 * 0.5
    sqrt2_over_sqrt_pi_input3 = input3 * 0.7071067811865476
    erf_result = tl.extra.cuda.libdevice.erf(sqrt2_over_sqrt_pi_input3)
    one_plus_erf = erf_result + 1.0
    gelu_approx = half_input3 * one_plus_erf
    product3 = input4 * input5
    difference = product3 - input6
    product4 = difference * input1
    product5 = product4 * product4
    product6 = product5 * input1
    small_constant = 2.869605142332415e-05
    product7 = product6 * small_constant
    sum1 = product2 + gelu_approx * product7
    product8 = -product7 * input5
    product9 = input4 * input1
    product10 = product9 * small_constant
    sum2 = product8 - product10
    sum3 = sum1 + sum2
    half_one_plus_erf = one_plus_erf * 0.5
    square_input3 = input3 * input3
    negative_half_square_input3 = square_input3 * -0.5
    exp_result = tl.math.exp(negative_half_square_input3)
    sqrt2_over_pi = 0.3989422804014327
    product11 = exp_result * sqrt2_over_pi
    product12 = input3 * product11
    sum4 = half_one_plus_erf + product12
    final_result = sum3 * sum4

    # Store result
    tl.store(in_out_ptr0 + (xindex), final_result, xmask)