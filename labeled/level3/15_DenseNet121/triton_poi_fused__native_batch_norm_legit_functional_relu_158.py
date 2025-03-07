# From: 15_DenseNet121

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_158poi_fused__native_batch_norm_legit_functional_relu_158(
    input_ptr, mean_ptr, variance_ptr, scale_ptr, bias_ptr, output_ptr, num_elements, BLOCK_SIZE: tl.constexpr
):
    num_elements = 297920
    block_offset = tl.program_id(0) * BLOCK_SIZE
    indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = indices < num_elements
    linear_index = indices
    channel_index = (indices // 49) % 608

    input_data = tl.load(input_ptr + (linear_index), mask)
    mean = tl.load(mean_ptr + (channel_index), mask, eviction_policy='evict_last')
    variance = tl.load(variance_ptr + (channel_index), mask, eviction_policy='evict_last')
    scale = tl.load(scale_ptr + (channel_index), mask, eviction_policy='evict_last')
    bias = tl.load(bias_ptr + (channel_index), mask, eviction_policy='evict_last')

    normalized_data = input_data - mean
    variance_epsilon = 1e-05
    normalized_variance = variance / 490.0 + variance_epsilon
    inv_std_dev = tl.extra.cuda.libdevice.rsqrt(normalized_variance)
    scaled_data = normalized_data * inv_std_dev * scale
    output_data = scaled_data + bias

    relu_output = tl.full([1], 0, tl.int32)
    relu_applied = triton_helpers.maximum(relu_output, output_data)
    tl.store(output_ptr + (linear_index), relu_applied, mask)