# From: 22_Matmul_Scale_ResidualAdd_Clamp_LogSumExp_Mish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_clamp_exp_fill_ge_le_logical_and_logsumexp_mish_mul_sigmoid_sub_where_0(
    in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    block_index = xindex // 1024
    element_index = xindex
    input_value0 = tl.load(in_ptr0 + (block_index), xmask, eviction_policy='evict_last')
    input_value1 = tl.load(in_ptr1 + (block_index), xmask, eviction_policy='evict_last')
    output_value = tl.load(in_out_ptr0 + (element_index), xmask)
    
    threshold = 20.0
    is_greater_than_threshold = input_value1 > threshold
    exp_value = tl.math.exp(input_value1)
    log1p_exp_value = tl.extra.cuda.libdevice.log1p(exp_value)
    logsumexp_value = tl.where(is_greater_than_threshold, input_value1, log1p_exp_value)
    
    tanh_value = tl.extra.cuda.libdevice.tanh(logsumexp_value)
    mish_product = input_value1 * tanh_value
    scaled_input0 = input_value0 * mish_product
    product_input0_input1 = input_value0 * input_value1
    
    sigmoid_value = tl.sigmoid(input_value1)
    sigmoid_product = input_value1 * sigmoid_value
    tanh_squared = tanh_value * tanh_value
    one_minus_tanh_squared = 1.0 - tanh_squared
    mish_addition = sigmoid_product * one_minus_tanh_squared
    mish_result = tanh_value + mish_addition
    
    scaled_product = product_input0_input1 * mish_result
    combined_result = scaled_input0 + scaled_product
    
    scale_factor = 2.0
    scaled_output = output_value * scale_factor
    doubled_scaled_output = scaled_output + scaled_output
    
    clamp_min = -10.0
    clamp_max = 10.0
    clamped_value = triton_helpers.maximum(doubled_scaled_output, clamp_min)
    clamped_value = triton_helpers.minimum(clamped_value, clamp_max)
    
    adjusted_value = clamped_value - input_value1
    exp_adjusted_value = tl.math.exp(adjusted_value)
    final_product = combined_result * exp_adjusted_value
    
    is_within_clamp_range = (doubled_scaled_output >= clamp_min) & (doubled_scaled_output <= clamp_max)
    zero_value = 0.0
    conditional_result = tl.where(is_within_clamp_range, final_product, zero_value)
    
    doubled_conditional_result = conditional_result + conditional_result
    final_scaled_result = doubled_conditional_result * scale_factor
    
    tl.store(in_out_ptr0 + (element_index), final_scaled_result, xmask)