# From: 97_Matmul_BatchNorm_BiasAdd_Divide_Swish

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_add_div_mul_sigmoid_1(
    in_out_ptr0, in_ptr_mean, in_ptr_var, in_ptr_gamma, in_ptr_beta, in_ptr_bias, in_ptr_scale, xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 512
    input_value = tl.load(in_ptr_mean + (x2), None)
    mean_value = tl.load(in_ptr_var + (x0), None, eviction_policy='evict_last')
    var_value = tl.load(in_ptr_gamma + (x0), None, eviction_policy='evict_last')
    gamma_value = tl.load(in_ptr_beta + (x0), None, eviction_policy='evict_last')
    beta_value = tl.load(in_ptr_bias + (x0), None, eviction_policy='evict_last')
    scale_value = tl.load(in_ptr_scale + (0))
    broadcast_scale = tl.broadcast_to(scale_value, [XBLOCK])
    
    normalized_value = input_value - mean_value
    scaled_variance = normalized_value * var_value
    scaled_gamma = scaled_variance * gamma_value
    shifted_beta = scaled_gamma + beta_value
    biased_output = shifted_beta + broadcast_scale
    
    one = 1.0
    scaled_output = biased_output * one
    sigmoid_output = tl.sigmoid(scaled_output)
    swish_output = scaled_output * sigmoid_output
    
    tl.store(in_out_ptr0 + (x2), swish_output, None)