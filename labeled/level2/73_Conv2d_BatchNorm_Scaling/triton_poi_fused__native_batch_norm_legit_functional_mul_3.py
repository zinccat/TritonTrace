# From: 73_Conv2d_BatchNorm_Scaling

import triton
import triton.language as tl


@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_mul_3(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_bias, input_ptr_input, 
    output_ptr, num_elements, XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 900) % 16
    
    mean = tl.load(input_ptr_mean + (x3), None)
    variance = tl.load(input_ptr_var + (x1), None, eviction_policy='evict_last')
    scale = tl.load(input_ptr_scale + (x1), None, eviction_policy='evict_last')
    bias = tl.load(input_ptr_bias + (x1), None, eviction_policy='evict_last')
    input_data = tl.load(input_ptr_input + (x3), None)
    
    normalized_input = input_data - mean
    variance_factor = 115200.0
    normalized_variance = variance / variance_factor
    epsilon = 1e-05
    adjusted_variance = normalized_variance + epsilon
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(adjusted_variance)
    scaled_input = normalized_input * inv_sqrt_variance
    scaled_and_shifted_input = scaled_input * scale
    output_data = scaled_and_shifted_input + bias
    final_output = output_data * 2.0
    
    tl.store(output_ptr + (x3), final_output, None)