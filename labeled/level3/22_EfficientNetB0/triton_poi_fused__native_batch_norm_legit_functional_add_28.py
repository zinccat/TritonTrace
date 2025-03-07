# From: 22_EfficientNetB0

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_add_28poi_fused__native_batch_norm_legit_functional_add_28(
    input_ptr, mean_ptr, variance_ptr, weight_ptr, bias_ptr, input_data_ptr, output_ptr, xnumel, XBLOCK: tl.constexpr
):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 24)
    
    input_data = tl.load(input_ptr + (x2), xmask)
    mean = tl.load(mean_ptr + (x0), xmask, eviction_policy='evict_last')
    variance = tl.load(variance_ptr + (x0), xmask, eviction_policy='evict_last')
    weight = tl.load(weight_ptr + (x0), xmask, eviction_policy='evict_last')
    bias = tl.load(bias_ptr + (x0), xmask, eviction_policy='evict_last')
    input_data_offset = tl.load(input_data_ptr + (x2), xmask)
    
    normalized_data = input_data - mean
    variance_epsilon = 1e-05
    normalized_variance = variance / 31360.0
    adjusted_variance = normalized_variance + variance_epsilon
    inv_std_dev = tl.extra.cuda.libdevice.rsqrt(adjusted_variance)
    scaled_data = normalized_data * inv_std_dev
    weighted_data = scaled_data * weight
    biased_data = weighted_data + bias
    output_data = biased_data + input_data_offset
    
    tl.store(output_ptr + (x2), output_data, xmask)