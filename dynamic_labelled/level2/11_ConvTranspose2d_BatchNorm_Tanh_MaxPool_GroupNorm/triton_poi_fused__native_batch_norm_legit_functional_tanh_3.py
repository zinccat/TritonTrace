# From: 11_ConvTranspose2d_BatchNorm_Tanh_MaxPool_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_tanh_3poi_fused__native_batch_norm_legit_functional_tanh_3(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_bias, input_ptr_x, 
    output_ptr, kernel_size, num_elements, XBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x3 = x_index
    x1 = ((x_index // 4096) % 64)
    
    input_mean = tl.load(input_ptr_mean + (x3), None)
    input_var = tl.load(input_ptr_var + (x1), None, eviction_policy='evict_last')
    input_scale = tl.load(input_ptr_scale + (x1), None, eviction_policy='evict_last')
    input_bias = tl.load(input_ptr_bias + (x1), None, eviction_policy='evict_last')
    input_x = tl.load(input_ptr_x + (x3), None)
    
    normalized_input = input_x - input_mean
    variance_factor = 4096 * kernel_size
    variance_factor_float = variance_factor.to(tl.float32)
    normalized_variance = input_var / variance_factor_float
    epsilon = 1e-05
    adjusted_variance = normalized_variance + epsilon
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(adjusted_variance)
    
    scaled_input = normalized_input * inv_sqrt_variance
    scaled_and_shifted_input = scaled_input * input_scale
    biased_input = scaled_and_shifted_input + input_bias
    tanh_output = tl.extra.cuda.libdevice.tanh(biased_input)
    
    tl.store(output_ptr + (x3), tanh_output, None)