# From: 24_EfficientNetB2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_32poi_fused__native_batch_norm_legit_functional_relu_32(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_bias, output_ptr, xnumel, XBLOCK: tl.constexpr
):
    xnumel = 2304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 1152)
    
    input_value = tl.load(input_ptr_mean + (x2), xmask)
    mean_value = tl.load(input_ptr_mean + (x0), xmask, eviction_policy='evict_last')
    var_value = tl.load(input_ptr_var + (x0), xmask, eviction_policy='evict_last')
    var_value_2 = tl.load(input_ptr_var + (1152 + x0), xmask, eviction_policy='evict_last')
    scale_value = tl.load(input_ptr_scale + (x0), xmask, eviction_policy='evict_last')
    bias_value = tl.load(input_ptr_bias + (x0), xmask, eviction_policy='evict_last')
    
    normalized_value = input_value - mean_value
    var_diff = var_value - mean_value
    var_diff_squared = var_diff * var_diff
    var_diff_2 = var_value_2 - mean_value
    var_diff_2_squared = var_diff_2 * var_diff_2
    var_sum = var_diff_squared + var_diff_2_squared
    var_sum_div_2 = var_sum / 2.0
    epsilon = 1e-05
    var_sum_div_2_plus_epsilon = var_sum_div_2 + epsilon
    inv_sqrt_var = tl.extra.cuda.libdevice.rsqrt(var_sum_div_2_plus_epsilon)
    scaled_normalized_value = normalized_value * inv_sqrt_var
    scaled_value = scaled_normalized_value * scale_value
    biased_value = scaled_value + bias_value
    
    zero_tensor = tl.full([1], 0, tl.int32)
    relu_output = triton_helpers.maximum(zero_tensor, biased_value)
    
    tl.store(output_ptr + (x2), relu_output, xmask)