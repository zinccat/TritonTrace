# From: 94_Gemm_BiasAdd_Hardtanh_Mish_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_fill_hardtanh_hardtanh_backward_mish_mul_native_group_norm_backward_sigmoid_sub_1(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel

    # Load inputs
    input0 = tl.load(in_ptr0 + xindex, xmask)
    input1 = tl.load(in_ptr1 + (xindex // 32), xmask, eviction_policy='evict_last')
    input2 = tl.load(in_ptr2 + (xindex % 1024), xmask, eviction_policy='evict_last')
    bias = tl.load(in_ptr3 + xindex, xmask)
    input4 = tl.load(in_ptr4 + (xindex % 1024), xmask, eviction_policy='evict_last')
    input5 = tl.load(in_ptr5 + (xindex // 32), xmask, eviction_policy='evict_last')
    input6 = tl.load(in_ptr6 + (xindex // 32), xmask, eviction_policy='evict_last')
    input7 = tl.load(in_ptr7 + (xindex // 32), xmask, eviction_policy='evict_last')

    # Intermediate calculations
    product1 = input1 * input2
    product2 = input0 * product1
    sum_bias_input4 = bias + input4
    neg_one = -1.0
    clamped_value = triton_helpers.maximum(sum_bias_input4, neg_one)
    one = 1.0
    min_clamped = triton_helpers.minimum(clamped_value, one)
    max_value = 20.0
    is_greater_than_max = min_clamped > max_value
    exp_value = tl.math.exp(min_clamped)
    log1p_value = tl.extra.cuda.libdevice.log1p(exp_value)
    tanh_value = tl.where(is_greater_than_max, min_clamped, log1p_value)
    hardtanh_value = min_clamped * tanh_value

    # GroupNorm backward
    groupnorm_diff = input5 * input6 - input7
    groupnorm_scaled = groupnorm_diff * input1
    groupnorm_cubed = groupnorm_scaled * groupnorm_scaled * groupnorm_scaled
    scale_factor = 0.03125
    groupnorm_scaled_factor = groupnorm_cubed * scale_factor
    hardtanh_scaled = hardtanh_value * groupnorm_scaled_factor
    output = product2 + hardtanh_scaled

    # GroupNorm gradient
    groupnorm_product = input5 * input6
    groupnorm_diff2 = groupnorm_product - input7
    groupnorm_scaled2 = groupnorm_diff2 * input1
    groupnorm_scaled_cubed = groupnorm_scaled2 * groupnorm_scaled2 * groupnorm_scaled2
    groupnorm_scaled_cubed_scaled = groupnorm_scaled_cubed * scale_factor
    neg_groupnorm_scaled_cubed_scaled = -groupnorm_scaled_cubed_scaled
    groupnorm_scaled_cubed_scaled_input6 = neg_groupnorm_scaled_cubed_scaled * input6
    groupnorm_scaled_input1 = input5 * input1
    groupnorm_scaled_input1_scaled = groupnorm_scaled_input1 * scale_factor
    groupnorm_gradient = groupnorm_scaled_cubed_scaled_input6 - groupnorm_scaled_input1_scaled
    final_output = output + groupnorm_gradient

    # Mish backward
    sigmoid_value = tl.sigmoid(min_clamped)
    mish_derivative = min_clamped * sigmoid_value
    tanh_squared = tanh_value * tanh_value
    one_minus_tanh_squared = one - tanh_squared
    mish_grad = mish_derivative * one_minus_tanh_squared
    mish_output = tanh_value + mish_grad
    final_mish_output = final_output * mish_output

    # Hardtanh backward
    is_less_than_neg_one = sum_bias_input4 <= neg_one
    is_greater_than_one = sum_bias_input4 >= one
    is_out_of_bounds = is_less_than_neg_one | is_greater_than_one
    zero = 0.0
    hardtanh_output = tl.where(is_out_of_bounds, zero, final_mish_output)

    # Store results
    tl.store(in_out_ptr0 + xindex, final_mish_output, xmask)
    tl.store(out_ptr0 + xindex, hardtanh_output, xmask)