# From: 9_ResNet18

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_add_relu_21poi_fused__native_batch_norm_legit_functional_add_relu_21(
    in_out_ptr, input_ptr, mean_ptr, inv_std_ptr, weight_ptr, bias_ptr, input_ptr2, running_mean_ptr, running_var_ptr, weight_ptr2, bias_ptr2, xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    
    input_value = tl.load(input_ptr + (x2), None)
    mean_value = tl.load(mean_ptr + (x0), None, eviction_policy='evict_last')
    inv_std_value = tl.load(inv_std_ptr + (x0), None, eviction_policy='evict_last')
    weight_value = tl.load(weight_ptr + (x0), None, eviction_policy='evict_last')
    bias_value = tl.load(bias_ptr + (x0), None, eviction_policy='evict_last')
    
    input_value2 = tl.load(input_ptr2 + (x2), None)
    running_mean_value = tl.load(running_mean_ptr + (x0), None, eviction_policy='evict_last')
    running_var_value = tl.load(running_var_ptr + (x0), None, eviction_policy='evict_last')
    weight_value2 = tl.load(weight_ptr2 + (x0), None, eviction_policy='evict_last')
    bias_value2 = tl.load(bias_ptr2 + (x0), None, eviction_policy='evict_last')
    
    normalized_input = input_value - mean_value
    scale_factor = 1568.0
    inv_std_adjusted = inv_std_value / scale_factor
    epsilon = 1e-05
    inv_std_adjusted_epsilon = inv_std_adjusted + epsilon
    inv_std_adjusted_epsilon_rsqrt = tl.extra.cuda.libdevice.rsqrt(inv_std_adjusted_epsilon)
    scaled_input = normalized_input * inv_std_adjusted_epsilon_rsqrt
    weighted_input = scaled_input * weight_value
    biased_input = weighted_input + bias_value
    
    normalized_input2 = input_value2 - running_mean_value
    inv_std_adjusted2 = running_var_value / scale_factor
    inv_std_adjusted2_epsilon = inv_std_adjusted2 + epsilon
    inv_std_adjusted2_epsilon_rsqrt = tl.extra.cuda.libdevice.rsqrt(inv_std_adjusted2_epsilon)
    scaled_input2 = normalized_input2 * inv_std_adjusted2_epsilon_rsqrt
    weighted_input2 = scaled_input2 * weight_value2
    biased_input2 = weighted_input2 + bias_value2
    
    fused_output = biased_input + biased_input2
    zero_tensor = tl.full([1], 0, tl.int32)
    relu_output = triton_helpers.maximum(zero_tensor, fused_output)
    
    tl.store(in_out_ptr + (x2), relu_output, None)