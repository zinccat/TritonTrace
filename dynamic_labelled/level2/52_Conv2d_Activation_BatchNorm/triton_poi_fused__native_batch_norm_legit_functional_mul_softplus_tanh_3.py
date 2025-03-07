# From: 52_Conv2d_Activation_BatchNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_mul_softplus_tanh_3poi_fused__native_batch_norm_legit_functional_mul_softplus_tanh_3(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_shift, input_ptr_bias, 
    output_ptr, kernel_size_0, kernel_size_1, kernel_size_2, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    element_index = index
    channel_index = (index // kernel_size_0) % 16

    mean = tl.load(input_ptr_mean + (element_index), mask, eviction_policy='evict_last')
    variance = tl.load(input_ptr_var + (channel_index), mask, eviction_policy='evict_last')
    scale = tl.load(input_ptr_scale + (channel_index), mask, eviction_policy='evict_last')
    shift = tl.load(input_ptr_shift + (channel_index), mask, eviction_policy='evict_last')
    bias = tl.load(input_ptr_bias + (channel_index), mask, eviction_policy='evict_last')

    threshold = 20.0
    is_greater_than_threshold = mean > threshold
    exp_mean = tl.math.exp(mean)
    log1p_exp_mean = tl.extra.cuda.libdevice.log1p(exp_mean)
    softplus_mean = tl.where(is_greater_than_threshold, mean, log1p_exp_mean)
    tanh_softplus = tl.extra.cuda.libdevice.tanh(softplus_mean)
    softplus_tanh_product = tanh_softplus * mean

    normalized_value = softplus_tanh_product - mean
    normalization_factor = 4 * kernel_size_1 + kernel_size_1 * kernel_size_2 * kernel_size_2 + ((-4) * kernel_size_1 * kernel_size_2)
    normalization_factor_float = normalization_factor.to(tl.float32)
    variance_normalized = variance / normalization_factor_float
    epsilon = 1e-05
    variance_normalized_epsilon = variance_normalized + epsilon
    rsqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_normalized_epsilon)
    scaled_normalized_value = normalized_value * rsqrt_variance
    scaled_normalized_value_scaled = scaled_normalized_value * scale
    batch_norm_output = scaled_normalized_value_scaled + shift + bias

    tl.store(output_ptr + (element_index), batch_norm_output, mask)