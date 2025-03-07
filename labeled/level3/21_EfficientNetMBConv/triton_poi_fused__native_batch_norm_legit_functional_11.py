# From: 21_EfficientNetMBConv

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_11poi_fused__native_batch_norm_legit_functional_11(
    input_ptr_mean, input_ptr_var, input_ptr_beta, input_ptr_gamma, input_ptr_input, 
    output_ptr, num_elements_y, num_elements_x, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr
):
    num_elements_y = 1920
    num_elements_x = 12544
    y_offset = tl.program_id(1) * YBLOCK
    y_index = y_offset + tl.arange(0, YBLOCK)[None, :]
    y_mask = y_index < num_elements_y
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < num_elements_x
    x2 = x_index
    y0 = (y_index % 192)
    y1 = y_index // 192
    y3 = y_index
    input_data = tl.load(input_ptr_input + (y0 + 192 * x2 + 2408448 * y1), xmask & ymask, eviction_policy='evict_last')
    mean = tl.load(input_ptr_mean + (y0), ymask, eviction_policy='evict_last')
    variance = tl.load(input_ptr_var + (y0), ymask, eviction_policy='evict_last')
    beta = tl.load(input_ptr_beta + (y0), ymask, eviction_policy='evict_last')
    gamma = tl.load(input_ptr_gamma + (y0), ymask, eviction_policy='evict_last')
    
    normalized_data = input_data - mean
    variance_scale = 125440.0
    epsilon = 1e-05
    variance_adjusted = variance / variance_scale
    variance_adjusted_epsilon = variance_adjusted + epsilon
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_adjusted_epsilon)
    scaled_data = normalized_data * inv_sqrt_variance
    scaled_gamma = scaled_data * gamma
    output_data = scaled_gamma + beta
    
    tl.store(output_ptr + (x2 + 12544 * y3), output_data, xmask & ymask)