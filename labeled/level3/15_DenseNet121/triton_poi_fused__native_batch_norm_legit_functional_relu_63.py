# From: 15_DenseNet121

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_63poi_fused__native_batch_norm_legit_functional_relu_63(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_shift, input_ptr_input, 
    output_ptr, num_elements, BLOCK_SIZE: tl.constexpr
):
    num_elements = 3512320
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    flat_index = block_indices
    batch_index = (block_indices // 784) % 448

    mean_value = tl.load(input_ptr_mean + (flat_index), valid_mask)
    variance_value = tl.load(input_ptr_var + (batch_index), valid_mask, eviction_policy='evict_last')
    scale_value = tl.load(input_ptr_scale + (batch_index), valid_mask, eviction_policy='evict_last')
    shift_value = tl.load(input_ptr_shift + (batch_index), valid_mask, eviction_policy='evict_last')
    input_value = tl.load(input_ptr_input + (flat_index), valid_mask)

    normalized_value = input_value - mean_value
    variance_adjusted = 7840.0
    epsilon = 1e-05
    variance_with_epsilon = variance_value / variance_adjusted + epsilon
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_with_epsilon)
    scaled_normalized_value = normalized_value * inv_sqrt_variance
    scaled_shifted_value = scaled_normalized_value * scale_value
    output_value = scaled_shifted_value + shift_value

    zero_tensor = tl.full([1], 0, tl.int32)
    relu_output = triton_helpers.maximum(zero_tensor, output_value)
    tl.store(output_ptr + (flat_index), relu_output, valid_mask)