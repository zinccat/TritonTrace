# From: 11_ConvTranspose2d_BatchNorm_Tanh_MaxPool_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_tanh_3(
    input_ptr, mean_ptr, variance_ptr, weight_ptr, bias_ptr, output_ptr, kernel_size, num_elements, XBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x3 = x_index
    x1 = ((x_index // 4096) % 64)
    
    input_value = tl.load(input_ptr + (x3), None)
    mean_value = tl.load(mean_ptr + (x1), None, eviction_policy='evict_last')
    variance_value = tl.load(variance_ptr + (x1), None, eviction_policy='evict_last')
    weight_value = tl.load(weight_ptr + (x1), None, eviction_policy='evict_last')
    bias_value = tl.load(bias_ptr + (x1), None, eviction_policy='evict_last')
    
    normalized_input = input_value - mean_value
    variance_scale = 4096 * kernel_size
    variance_scale_float = variance_scale.to(tl.float32)
    variance_adjusted = variance_value / variance_scale_float
    epsilon = 1e-05
    variance_adjusted_epsilon = variance_adjusted + epsilon
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_adjusted_epsilon)
    
    scaled_input = normalized_input * inv_sqrt_variance
    weighted_input = scaled_input * weight_value
    biased_input = weighted_input + bias_value
    tanh_output = tl.extra.cuda.libdevice.tanh(biased_input)
    
    tl.store(output_ptr + (x3), tanh_output, None)