# From: 22_Matmul_Scale_ResidualAdd_Clamp_LogSumExp_Mish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_clamp_exp_fill_ge_le_logical_and_logsumexp_mish_mul_sigmoid_sub_where_0poi_fused_add_clamp_exp_fill_ge_le_logical_and_logsumexp_mish_mul_sigmoid_sub_where_0(
    in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    block_id = xindex // 1024
    element_id = xindex
    input0 = tl.load(in_ptr0 + (block_id), xmask, eviction_policy='evict_last')
    input1 = tl.load(in_ptr1 + (block_id), xmask, eviction_policy='evict_last')
    output = tl.load(in_out_ptr0 + (element_id), xmask)
    
    threshold = 20.0
    exp_input1 = tl.math.exp(input1)
    log1p_exp_input1 = tl.extra.cuda.libdevice.log1p(exp_input1)
    logsumexp_input1 = tl.where(input1 > threshold, input1, log1p_exp_input1)
    
    tanh_logsumexp_input1 = tl.extra.cuda.libdevice.tanh(logsumexp_input1)
    mish_input1 = input1 * tanh_logsumexp_input1
    scaled_input0_mish = input0 * mish_input1
    scaled_input0_input1 = input0 * input1
    
    sigmoid_input1 = tl.sigmoid(input1)
    scaled_input1_sigmoid = input1 * sigmoid_input1
    tanh_squared = tanh_logsumexp_input1 * tanh_logsumexp_input1
    one_minus_tanh_squared = 1.0 - tanh_squared
    scaled_sigmoid_one_minus_tanh_squared = scaled_input1_sigmoid * one_minus_tanh_squared
    mish_addition = tanh_logsumexp_input1 + scaled_sigmoid_one_minus_tanh_squared
    scaled_input0_mish_addition = scaled_input0_input1 * mish_addition
    
    fused_output = scaled_input0_mish + scaled_input0_mish_addition
    
    scale_factor = 2.0
    scaled_output = output * scale_factor
    doubled_scaled_output = scaled_output + scaled_output
    
    clamp_min = -10.0
    clamp_max = 10.0
    clamped_output = triton_helpers.maximum(doubled_scaled_output, clamp_min)
    clamped_output = triton_helpers.minimum(clamped_output, clamp_max)
    
    exp_clamped_minus_input1 = tl.math.exp(clamped_output - input1)
    scaled_fused_output = fused_output * exp_clamped_minus_input1
    
    ge_clamp_min = doubled_scaled_output >= clamp_min
    le_clamp_max = doubled_scaled_output <= clamp_max
    within_clamp_range = ge_clamp_min & le_clamp_max
    
    zero = 0.0
    conditional_output = tl.where(within_clamp_range, scaled_fused_output, zero)
    doubled_conditional_output = conditional_output + conditional_output
    final_output = doubled_conditional_output * scale_factor
    
    tl.store(in_out_ptr0 + (element_id), final_output, xmask)