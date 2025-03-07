# From: 72_ConvTranspose3d_BatchNorm_AvgPool_AvgPool

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_4(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_bias, input_ptr_input, 
    output_ptr, kernel_size_0, kernel_size_1, kernel_size_2, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    input_index = index
    channel_index = ((index // kernel_size_0) % 16)
    
    mean = tl.load(input_ptr_mean + (input_index), mask, eviction_policy='evict_last')
    variance = tl.load(input_ptr_var + (channel_index), mask, eviction_policy='evict_last')
    scale = tl.load(input_ptr_scale + (channel_index), mask, eviction_policy='evict_last')
    bias = tl.load(input_ptr_bias + (channel_index), mask, eviction_policy='evict_last')
    input_data = tl.load(input_ptr_input + (input_index), mask, eviction_policy='evict_last')
    
    normalized_input = input_data - mean
    variance_adjustment = ((-1) * kernel_size_1) + ((-12) * kernel_size_1 * kernel_size_2 * kernel_size_2) + 6 * kernel_size_1 * kernel_size_2 + 8 * kernel_size_1 * kernel_size_2 * kernel_size_2 * kernel_size_2
    variance_adjustment_float = variance_adjustment.to(tl.float32)
    variance_normalized = variance / variance_adjustment_float
    epsilon = 1e-05
    variance_stabilized = variance_normalized + epsilon
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_stabilized)
    scaled_input = normalized_input * inv_sqrt_variance
    scaled_and_shifted_input = scaled_input * scale
    output_data = scaled_and_shifted_input + bias
    
    tl.store(output_ptr + (input_index), output_data, mask)