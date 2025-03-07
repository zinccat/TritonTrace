# From: 20_MobileNetV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_61poi_fused__native_batch_norm_legit_functional_61(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_bias, input_ptr_input, 
    output_ptr, xnumel, XBLOCK: tl.constexpr
):
    xnumel = 156800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 320)
    
    input_mean = tl.load(input_ptr_mean + (x2), xmask)
    input_var = tl.load(input_ptr_var + (x0), xmask, eviction_policy='evict_last')
    input_scale = tl.load(input_ptr_scale + (x0), xmask, eviction_policy='evict_last')
    input_bias = tl.load(input_ptr_bias + (x0), xmask, eviction_policy='evict_last')
    input_data = tl.load(input_ptr_input + (x2), xmask)
    
    normalized_data = input_data - input_mean
    variance_scale = 490.0
    normalized_variance = input_var / variance_scale
    epsilon = 1e-05
    adjusted_variance = normalized_variance + epsilon
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(adjusted_variance)
    scaled_data = normalized_data * inv_sqrt_variance
    scaled_and_shifted_data = scaled_data * input_scale
    output_data = scaled_and_shifted_data + input_bias
    
    tl.store(output_ptr + (x2), output_data, xmask)