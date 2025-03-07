# From: 24_EfficientNetB2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_30poi_fused__native_batch_norm_legit_functional_30(input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_bias, output_ptr, num_elements, BLOCK_SIZE : tl.constexpr):
    num_elements = 384
    block_offset = tl.program_id(0) * BLOCK_SIZE
    element_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = element_indices < num_elements
    global_indices = element_indices
    channel_indices = element_indices % 192

    input_data = tl.load(input_ptr_mean + (global_indices), valid_mask)
    mean = tl.load(input_ptr_mean + (channel_indices), valid_mask, eviction_policy='evict_last')
    var_mean = tl.load(input_ptr_mean + (channel_indices), valid_mask, eviction_policy='evict_last')
    var_mean_plus_one = tl.load(input_ptr_mean + (192 + channel_indices), valid_mask, eviction_policy='evict_last')
    scale = tl.load(input_ptr_scale + (channel_indices), valid_mask, eviction_policy='evict_last')
    bias = tl.load(input_ptr_bias + (channel_indices), valid_mask, eviction_policy='evict_last')

    normalized_data = input_data - mean
    var_diff = var_mean - mean
    var_diff_squared = var_diff * var_diff
    var_mean_plus_one_diff = var_mean_plus_one - mean
    var_mean_plus_one_diff_squared = var_mean_plus_one_diff * var_mean_plus_one_diff
    variance_sum = var_diff_squared + var_mean_plus_one_diff_squared
    variance_sum_divided = variance_sum / 2.0
    epsilon = 1e-05
    variance_sum_with_epsilon = variance_sum_divided + epsilon
    reciprocal_sqrt = tl.extra.cuda.libdevice.rsqrt(variance_sum_with_epsilon)
    normalized_scaled_data = normalized_data * reciprocal_sqrt
    scaled_data = normalized_scaled_data * scale
    output_data = scaled_data + bias

    tl.store(output_ptr + (global_indices), output_data, valid_mask)