# From: 19_ConvTranspose2d_GELU_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_gelu_gelu_backward_native_group_norm_backward_3poi_fused_gelu_gelu_backward_native_group_norm_backward_3(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x6 = xindex // 34848
    x7 = ((xindex // 4356) % 64)

    input0 = tl.load(in_ptr0 + (x4), xmask)
    input1 = tl.load(in_ptr1 + (x6), xmask, eviction_policy='evict_last')
    input2 = tl.load(in_ptr2 + (x7), xmask, eviction_policy='evict_last')
    input3 = tl.load(in_ptr3 + (x4), xmask)
    input4 = tl.load(in_ptr4 + (x6), xmask, eviction_policy='evict_last')
    input5 = tl.load(in_ptr5 + (x6), xmask, eviction_policy='evict_last')
    input6 = tl.load(in_ptr6 + (x6), xmask, eviction_policy='evict_last')

    product1 = input1 * input2
    product2 = input0 * product1
    half = 0.5
    half_input3 = input3 * half
    sqrt2_over_sqrt_pi = 0.7071067811865476
    sqrt2_over_sqrt_pi_input3 = input3 * sqrt2_over_sqrt_pi
    erf_result = tl.extra.cuda.libdevice.erf(sqrt2_over_sqrt_pi_input3)
    one = 1.0
    erf_plus_one = erf_result + one
    product3 = half_input3 * erf_plus_one

    product4 = input4 * input5
    difference = product4 - input6
    product5 = difference * input1
    product6 = product5 * product5
    product7 = product6 * product5
    small_constant = 2.869605142332415e-05
    product8 = product7 * small_constant
    sum1 = product2 + product8 * product3

    neg_small_constant = -small_constant
    product9 = neg_small_constant * input5
    product10 = input4 * input1
    product11 = product10 * small_constant
    sum2 = product9 - product11
    sum3 = sum1 + sum2

    erf_half = erf_plus_one * half
    square_input3 = input3 * input3
    neg_half = -0.5
    exp_result = tl.math.exp(square_input3 * neg_half)
    sqrt_pi_over_sqrt2 = 0.3989422804014327
    product12 = exp_result * sqrt_pi_over_sqrt2
    product13 = input3 * product12
    sum4 = erf_half + product13
    final_result = sum3 * sum4

    tl.store(in_out_ptr0 + (x4), final_result, xmask)