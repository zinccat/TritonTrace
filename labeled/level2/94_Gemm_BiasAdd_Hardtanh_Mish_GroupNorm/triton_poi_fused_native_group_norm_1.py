# From: 94_Gemm_BiasAdd_Hardtanh_Mish_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused_native_group_norm_1(input_ptr_mean, input_ptr_var, input_ptr_bn_mean, input_ptr_bn_var, input_ptr_bn_weight, input_ptr_bn_bias, output_ptr, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 1024
    input_mean = tl.load(input_ptr_mean + (x2), None)
    input_var = tl.load(input_ptr_var + (x0), None, eviction_policy='evict_last')
    bn_mean = tl.load(input_ptr_bn_mean + ((x2 // 32)), None, eviction_policy='evict_last')
    bn_var = tl.load(input_ptr_bn_var + ((x2 // 32)), None, eviction_policy='evict_last')
    bn_weight = tl.load(input_ptr_bn_weight + (x0), None, eviction_policy='evict_last')
    bn_bias = tl.load(input_ptr_bn_bias + (x0), None, eviction_policy='evict_last')
    
    normalized_input = input_mean + input_var
    lower_bound = -1.0
    upper_bound = 1.0
    clamped_input = triton_helpers.maximum(normalized_input, lower_bound)
    clamped_input = triton_helpers.minimum(clamped_input, upper_bound)
    
    threshold = 20.0
    is_exceeding_threshold = clamped_input > threshold
    exp_clamped_input = tl.math.exp(clamped_input)
    log1p_exp_clamped_input = tl.extra.cuda.libdevice.log1p(exp_clamped_input)
    mish_activation = tl.where(is_exceeding_threshold, clamped_input, log1p_exp_clamped_input)
    tanh_mish = tl.extra.cuda.libdevice.tanh(mish_activation)
    mish_output = clamped_input * tanh_mish
    
    normalized_output = mish_output - bn_mean
    bn_var_scaled = bn_var / 32.0
    epsilon = 1e-05
    variance_with_epsilon = bn_var_scaled + epsilon
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_with_epsilon)
    scaled_output = normalized_output * inv_sqrt_variance
    weighted_output = scaled_output * bn_weight
    final_output = weighted_output + bn_bias
    
    tl.store(output_ptr + (x2), final_output, None)