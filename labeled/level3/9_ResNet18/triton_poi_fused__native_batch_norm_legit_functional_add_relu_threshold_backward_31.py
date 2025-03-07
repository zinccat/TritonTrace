# From: 9_ResNet18

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_31poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_31(
    input_grad_output_ptr, input_mean_ptr, input_inv_std_ptr, input_weight_ptr, input_bias_ptr, input_grad_input_ptr, 
    output_grad_output_ptr, output_grad_input_ptr, xnumel, XBLOCK: tl.constexpr
):
    xnumel = 50176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 512)
    
    grad_output = tl.load(input_grad_output_ptr + (x2), xmask)
    mean = tl.load(input_mean_ptr + (x0), xmask, eviction_policy='evict_last')
    inv_std = tl.load(input_inv_std_ptr + (x0), xmask, eviction_policy='evict_last')
    weight = tl.load(input_weight_ptr + (x0), xmask, eviction_policy='evict_last')
    bias = tl.load(input_bias_ptr + (x0), xmask, eviction_policy='evict_last')
    grad_input = tl.load(input_grad_input_ptr + (x2), xmask)
    
    normalized_grad_output = grad_output - mean
    eps = 1e-05
    adjusted_inv_std = inv_std / 98.0 + eps
    inv_std_rsqrt = tl.extra.cuda.libdevice.rsqrt(adjusted_inv_std)
    scaled_grad_output = normalized_grad_output * inv_std_rsqrt
    weighted_grad_output = scaled_grad_output * weight
    biased_grad_output = weighted_grad_output + bias
    relu_grad_output = biased_grad_output + grad_input
    
    zero_tensor = tl.full([1], 0, tl.int32)
    relu_output = triton_helpers.maximum(zero_tensor, relu_grad_output)
    threshold = 0.0
    relu_output_mask = relu_output <= threshold
    
    tl.store(output_grad_output_ptr + (x2), relu_output, xmask)
    tl.store(output_grad_input_ptr + (x2), relu_output_mask, xmask)