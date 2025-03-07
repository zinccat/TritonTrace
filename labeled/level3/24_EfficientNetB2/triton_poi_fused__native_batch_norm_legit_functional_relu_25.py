# From: 24_EfficientNetB2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_25poi_fused__native_batch_norm_legit_functional_relu_25(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_bias, output_ptr, xnumel, XBLOCK: tl.constexpr
):
    xnumel = 1728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 864)
    
    input_value = tl.load(input_ptr_mean + (x2), xmask)
    mean_value = tl.load(input_ptr_mean + (x0), xmask, eviction_policy='evict_last')
    mean_value_repeated = tl.load(input_ptr_mean + (x0), xmask, eviction_policy='evict_last')
    mean_value_next = tl.load(input_ptr_mean + (864 + x0), xmask, eviction_policy='evict_last')
    scale_value = tl.load(input_ptr_scale + (x0), xmask, eviction_policy='evict_last')
    bias_value = tl.load(input_ptr_bias + (x0), xmask, eviction_policy='evict_last')
    
    centered_input = input_value - mean_value
    centered_input_repeated = mean_value_repeated - mean_value
    squared_centered_input_repeated = centered_input_repeated * centered_input_repeated
    centered_input_next = mean_value_next - mean_value
    squared_centered_input_next = centered_input_next * centered_input_next
    variance = squared_centered_input_repeated + squared_centered_input_next
    variance_divisor = 2.0
    variance_adjusted = variance / variance_divisor
    epsilon = 1e-05
    variance_epsilon_adjusted = variance_adjusted + epsilon
    inv_stddev = tl.extra.cuda.libdevice.rsqrt(variance_epsilon_adjusted)
    normalized_input = centered_input * inv_stddev
    scaled_input = normalized_input * scale_value
    biased_input = scaled_input + bias_value
    
    zero_tensor = tl.full([1], 0, tl.int32)
    relu_output = triton_helpers.maximum(zero_tensor, biased_input)
    
    tl.store(output_ptr + (x2), relu_output, xmask)