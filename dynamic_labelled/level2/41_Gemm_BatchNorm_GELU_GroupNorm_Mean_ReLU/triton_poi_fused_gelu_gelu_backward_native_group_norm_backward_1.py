# From: 41_Gemm_BatchNorm_GELU_GroupNorm_Mean_ReLU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_gelu_gelu_backward_native_group_norm_backward_1poi_fused_gelu_gelu_backward_native_group_norm_backward_1(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    xblock_div_1024 = xindex // 1024
    xblock_div_128 = xindex // 128
    xindex_mod_1024 = xindex % 1024
    xindex_full = xindex

    # Load data with eviction policy
    mask_flag = tl.load(in_ptr0 + (xblock_div_1024), xmask, eviction_policy='evict_last').to(tl.int1)
    input_data1 = tl.load(in_ptr1 + (xblock_div_1024), xmask, eviction_policy='evict_last')
    input_data2 = tl.load(in_ptr2 + (xblock_div_128), xmask, eviction_policy='evict_last')
    input_data3 = tl.load(in_ptr3 + (xindex_mod_1024), xmask, eviction_policy='evict_last')
    input_data4 = tl.load(in_ptr4 + (xindex_full), xmask)
    input_data5 = tl.load(in_ptr5 + (xblock_div_128), xmask, eviction_policy='evict_last')
    input_data6 = tl.load(in_ptr6 + (xblock_div_128), xmask, eviction_policy='evict_last')
    input_data7 = tl.load(in_ptr7 + (xblock_div_128), xmask, eviction_policy='evict_last')
    input_data8 = tl.load(in_ptr5 + (xindex_full // 128), xmask, eviction_policy='evict_last')
    input_data9 = tl.load(in_ptr6 + (xindex_full // 128), xmask, eviction_policy='evict_last')
    input_data10 = tl.load(in_ptr7 + (xindex_full // 128), xmask, eviction_policy='evict_last')
    input_data11 = tl.load(in_ptr2 + (xindex_full // 128), xmask, eviction_policy='evict_last')

    # Compute intermediate values
    zero_value = 0.0
    selected_input = tl.where(mask_flag, zero_value, input_data1)
    scale_factor = 0.0009765625
    scaled_input = selected_input * scale_factor
    product1 = input_data2 * input_data3
    scaled_product = scaled_input * product1
    half_value = 0.5
    half_input = input_data4 * half_value
    sqrt2_over2 = 0.7071067811865476
    sqrt2_input = input_data4 * sqrt2_over2
    erf_result = tl.extra.cuda.libdevice.erf(sqrt2_input)
    one_value = 1.0
    erf_plus_one = erf_result + one_value
    erf_scaled = half_input * erf_plus_one
    product2 = input_data5 * input_data6
    difference1 = product2 - input_data7
    scaled_difference = difference1 * input_data2
    scaled_difference_cubed = scaled_difference * scaled_difference * scaled_difference
    scale_cubed_factor = 0.0078125
    scaled_cubed_product = scaled_difference_cubed * scale_cubed_factor
    sum1 = erf_scaled * scaled_cubed_product
    final_sum1 = scaled_product + sum1
    product3 = input_data8 * input_data9
    difference2 = product3 - input_data10
    scaled_difference2 = difference2 * input_data11
    scaled_difference2_cubed = scaled_difference2 * scaled_difference2 * scaled_difference2
    scaled_difference2_cubed_factor = scaled_difference2_cubed * scale_cubed_factor
    neg_scaled_difference2_cubed = -scaled_difference2_cubed_factor
    neg_scaled_product = neg_scaled_difference2_cubed * input_data9
    product4 = input_data8 * input_data11
    scaled_product2 = product4 * scale_cubed_factor
    final_neg_product = neg_scaled_product - scaled_product2
    final_sum2 = final_sum1 + final_neg_product
    erf_half = erf_plus_one * half_value
    squared_input = input_data4 * input_data4
    neg_half = -0.5
    exp_component = tl.math.exp(squared_input * neg_half)
    sqrt2_pi = 0.3989422804014327
    gaussian_component = exp_component * sqrt2_pi
    scaled_gaussian = input_data4 * gaussian_component
    final_erf = erf_half + scaled_gaussian
    final_product = final_sum2 * final_erf
    tl.store(in_out_ptr0 + (xindex_full), final_product, xmask)