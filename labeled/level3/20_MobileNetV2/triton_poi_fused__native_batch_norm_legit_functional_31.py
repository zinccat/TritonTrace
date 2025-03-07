# From: 20_MobileNetV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_31poi_fused__native_batch_norm_legit_functional_31(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_bias, input_ptr_input, 
    output_ptr, xnumel, XBLOCK: tl.constexpr
):
    xnumel = 250880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 32)
    
    input_data = tl.load(input_ptr_input + (x2), xmask)
    mean = tl.load(input_ptr_mean + (x0), xmask, eviction_policy='evict_last')
    variance = tl.load(input_ptr_var + (x0), xmask, eviction_policy='evict_last')
    scale = tl.load(input_ptr_scale + (x0), xmask, eviction_policy='evict_last')
    bias = tl.load(input_ptr_bias + (x0), xmask, eviction_policy='evict_last')
    
    normalized_data = input_data - mean
    variance_adjusted = 7840.0
    epsilon = 1e-05
    variance_with_epsilon = variance / variance_adjusted + epsilon
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_with_epsilon)
    
    scaled_data = normalized_data * inv_sqrt_variance
    scaled_and_shifted_data = scaled_data * scale
    output_data = scaled_and_shifted_data + bias
    
    tl.store(output_ptr + (x2), output_data, xmask)