# From: 10_ResNet101

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_51poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_51(
    input_grad_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr, output_ptr, grad_output_ptr, grad_input_ptr, numel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 2048)
    
    input_grad = tl.load(input_grad_ptr + (x2), None)
    running_mean = tl.load(running_mean_ptr + (x0), None, eviction_policy='evict_last')
    running_var = tl.load(running_var_ptr + (x0), None, eviction_policy='evict_last')
    weight = tl.load(weight_ptr + (x0), None, eviction_policy='evict_last')
    bias = tl.load(bias_ptr + (x0), None, eviction_policy='evict_last')
    output = tl.load(output_ptr + (x2), None)
    
    normalized_grad = input_grad - running_mean
    variance_scale = 490.0
    normalized_variance = running_var / variance_scale
    epsilon = 1e-05
    adjusted_variance = normalized_variance + epsilon
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(adjusted_variance)
    scaled_grad = normalized_grad * inv_sqrt_variance
    weighted_grad = scaled_grad * weight
    biased_grad = weighted_grad + bias
    final_output = biased_grad + output
    
    zero_tensor = tl.full([1], 0, tl.int32)
    relu_output = triton_helpers.maximum(zero_tensor, final_output)
    threshold = 0.0
    relu_mask = relu_output <= threshold
    
    tl.store(grad_output_ptr + (x2), relu_output, None)
    tl.store(grad_input_ptr + (x2), relu_mask, None)