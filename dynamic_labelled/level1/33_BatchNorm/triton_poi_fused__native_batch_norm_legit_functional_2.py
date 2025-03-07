# From: 33_BatchNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_2poi_fused__native_batch_norm_legit_functional_2(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_shift, input_ptr_out, 
    output_ptr, kernel_size_0, kernel_size_1, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    mask = index < num_elements
    element_index = index
    channel_index = (index // kernel_size_0) % 64

    mean_value = tl.load(input_ptr_mean + (element_index), mask, eviction_policy='evict_last')
    variance_value = tl.load(input_ptr_var + (channel_index), mask, eviction_policy='evict_last')
    scale_value = tl.load(input_ptr_scale + (channel_index), mask, eviction_policy='evict_last')
    shift_value = tl.load(input_ptr_shift + (channel_index), mask, eviction_policy='evict_last')
    output_value = tl.load(input_ptr_out + (channel_index), mask, eviction_policy='evict_last')

    normalized_value = mean_value - variance_value
    num_elements_float = (kernel_size_0 * kernel_size_1).to(tl.float32)
    variance_normalized = variance_value / num_elements_float
    epsilon = 1e-05
    variance_stabilized = variance_normalized + epsilon
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_stabilized)

    scaled_value = normalized_value * inv_sqrt_variance
    scaled_and_shifted_value = scaled_value * scale_value
    final_output_value = scaled_and_shifted_value + shift_value

    tl.store(output_ptr + (element_index), final_output_value, mask)