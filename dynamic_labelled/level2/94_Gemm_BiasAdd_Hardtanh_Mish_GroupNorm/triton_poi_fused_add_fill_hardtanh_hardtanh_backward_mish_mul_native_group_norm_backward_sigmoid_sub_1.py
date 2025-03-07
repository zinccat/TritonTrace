# From: 94_Gemm_BiasAdd_Hardtanh_Mish_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_fill_hardtanh_hardtanh_backward_mish_mul_native_group_norm_backward_sigmoid_sub_1poi_fused_add_fill_hardtanh_hardtanh_backward_mish_mul_native_group_norm_backward_sigmoid_sub_1(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    x_mask = x_index < xnumel
    x4 = x_index
    x5 = x_index // 32
    x3 = x_index % 1024

    input0 = tl.load(in_ptr0 + (x4), x_mask)
    input1 = tl.load(in_ptr1 + (x5), x_mask, eviction_policy='evict_last')
    input2 = tl.load(in_ptr2 + (x3), x_mask, eviction_policy='evict_last')
    bias = tl.load(in_ptr3 + (x4), x_mask)
    input4 = tl.load(in_ptr4 + (x3), x_mask, eviction_policy='evict_last')
    input5 = tl.load(in_ptr5 + (x5), x_mask, eviction_policy='evict_last')
    input6 = tl.load(in_ptr6 + (x5), x_mask, eviction_policy='evict_last')
    input7 = tl.load(in_ptr7 + (x5), x_mask, eviction_policy='evict_last')
    input5_2 = tl.load(in_ptr5 + (x4 // 32), x_mask, eviction_policy='evict_last')
    input6_2 = tl.load(in_ptr6 + (x4 // 32), x_mask, eviction_policy='evict_last')
    input7_2 = tl.load(in_ptr7 + (x4 // 32), x_mask, eviction_policy='evict_last')
    input1_2 = tl.load(in_ptr1 + (x4 // 32), x_mask, eviction_policy='evict_last')

    product1 = input1 * input2
    product2 = input0 * product1
    sum1 = bias + input4
    neg_one = -1.0
    clamped_max = triton_helpers.maximum(sum1, neg_one)
    one = 1.0
    clamped_min = triton_helpers.minimum(clamped_max, one)
    max_value = 20.0
    is_greater_than_max = clamped_min > max_value
    exp_clamped_min = tl.math.exp(clamped_min)
    log1p_exp = tl.extra.cuda.libdevice.log1p(exp_clamped_min)
    tanh_input = tl.where(is_greater_than_max, clamped_min, log1p_exp)
    tanh_result = tl.extra.cuda.libdevice.tanh(tanh_input)
    product3 = clamped_min * tanh_result

    product4 = input5 * input6
    difference1 = product4 - input7
    product5 = difference1 * input1
    product6 = product5 * input1
    product7 = product6 * input1
    scale_factor = 0.03125
    scaled_product7 = product7 * scale_factor
    product8 = product3 * scaled_product7
    result1 = product2 + product8

    product9 = input5_2 * input6_2
    difference2 = product9 - input7_2
    product10 = difference2 * input1_2
    product11 = product10 * input1_2
    product12 = product11 * input1_2
    product13 = product12 * scale_factor
    neg_product13 = -product13
    product14 = neg_product13 * input6_2
    product15 = input5_2 * input1_2
    product16 = product15 * scale_factor
    difference3 = product14 - product16
    result2 = result1 + difference3

    sigmoid_result = tl.sigmoid(clamped_min)
    product17 = clamped_min * sigmoid_result
    tanh_squared = tanh_result * tanh_result
    difference4 = one - tanh_squared
    product18 = product17 * difference4
    result3 = tanh_result + product18
    final_result = result2 * result3

    is_less_than_neg_one = sum1 <= neg_one
    is_greater_than_one = sum1 >= one
    is_out_of_bounds = is_less_than_neg_one | is_greater_than_one
    zero = 0.0
    clamped_result = tl.where(is_out_of_bounds, zero, final_result)

    tl.store(in_out_ptr0 + (x4), final_result, x_mask)
    tl.store(out_ptr0 + (x4), clamped_result, x_mask)