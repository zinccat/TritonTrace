# From: 33_BatchNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_2(input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_shift, input_ptr_out, output_ptr, kernel_size_0, kernel_size_1, num_elements, XBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    element_index = index
    channel_index = (index // kernel_size_0) % 64
    
    mean = tl.load(input_ptr_mean + (element_index), mask, eviction_policy='evict_last')
    variance = tl.load(input_ptr_var + (channel_index), mask, eviction_policy='evict_last')
    scale = tl.load(input_ptr_scale + (channel_index), mask, eviction_policy='evict_last')
    shift = tl.load(input_ptr_shift + (channel_index), mask, eviction_policy='evict_last')
    output = tl.load(input_ptr_out + (element_index), mask, eviction_policy='evict_last')
    
    centered_input = output - mean
    num_elements_float = (kernel_size_0 * kernel_size_1).to(tl.float32)
    normalized_variance = variance / num_elements_float
    epsilon = 1e-05
    adjusted_variance = normalized_variance + epsilon
    inv_stddev = tl.extra.cuda.libdevice.rsqrt(adjusted_variance)
    
    normalized_output = centered_input * inv_stddev
    scaled_output = normalized_output * scale
    shifted_output = scaled_output + shift
    
    tl.store(output_ptr + (element_index), shifted_output, mask)