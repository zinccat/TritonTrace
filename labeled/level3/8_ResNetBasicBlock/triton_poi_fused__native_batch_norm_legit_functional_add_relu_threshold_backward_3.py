# From: 8_ResNetBasicBlock

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_3poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_3(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, 
    out_ptr0, xnumel, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 50176) % 64)
    
    input_value = tl.load(in_ptr0 + (x3), None)
    mean = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    variance = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    weight = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    bias = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    grad_output = tl.load(in_ptr5 + (x3), None)
    grad_mean = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    grad_variance = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    running_mean = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    running_variance = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    
    centered_input = input_value - mean
    variance_scale = 501760.0
    normalized_variance = variance / variance_scale
    epsilon = 1e-05
    adjusted_variance = normalized_variance + epsilon
    inv_std_dev = tl.extra.cuda.libdevice.rsqrt(adjusted_variance)
    normalized_input = centered_input * inv_std_dev
    scaled_input = normalized_input * weight
    batch_norm_output = scaled_input + bias
    
    grad_input_centered = grad_output - grad_mean
    normalized_grad_variance = grad_variance / variance_scale
    adjusted_grad_variance = normalized_grad_variance + epsilon
    inv_std_dev_grad = tl.extra.cuda.libdevice.rsqrt(adjusted_grad_variance)
    normalized_grad_input = grad_input_centered * inv_std_dev_grad
    scaled_grad_input = normalized_grad_input * running_mean
    grad_batch_norm_output = scaled_grad_input + running_variance
    
    combined_output = batch_norm_output + grad_batch_norm_output
    zero_tensor = tl.full([1], 0, tl.int32)
    relu_output = triton_helpers.maximum(zero_tensor, combined_output)
    threshold = 0.0
    relu_output_leq_threshold = relu_output <= threshold
    
    tl.store(in_out_ptr0 + (x3), relu_output, None)
    tl.store(out_ptr0 + (x3), relu_output_leq_threshold, None)