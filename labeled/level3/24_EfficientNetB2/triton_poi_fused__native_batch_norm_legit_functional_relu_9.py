# From: 24_EfficientNetB2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_9poi_fused__native_batch_norm_legit_functional_relu_9(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_shift, input_ptr_bias, 
    output_ptr, num_elements, BLOCK_SIZE: tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    global_indices = block_indices
    channel_indices = block_indices % 96

    mean_value = tl.load(input_ptr_mean + (global_indices), None)
    variance_value = tl.load(input_ptr_var + (channel_indices), None, eviction_policy='evict_last')
    scale_value = tl.load(input_ptr_scale + (channel_indices), None, eviction_policy='evict_last')
    shift_value = tl.load(input_ptr_shift + (channel_indices), None, eviction_policy='evict_last')
    bias_value = tl.load(input_ptr_bias + (channel_indices), None, eviction_policy='evict_last')

    normalized_value = mean_value - variance_value
    variance_normalized = 25088.0
    epsilon = 1e-05
    variance_adjusted = variance_value / variance_normalized
    variance_adjusted_epsilon = variance_adjusted + epsilon
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_adjusted_epsilon)
    scaled_normalized_value = normalized_value * inv_sqrt_variance
    scaled_value = scaled_normalized_value * scale_value
    shifted_value = scaled_value + shift_value
    relu_output = tl.full([1], 0, tl.int32)
    relu_applied = triton_helpers.maximum(relu_output, shifted_value)

    tl.store(output_ptr + (global_indices), relu_applied, None)