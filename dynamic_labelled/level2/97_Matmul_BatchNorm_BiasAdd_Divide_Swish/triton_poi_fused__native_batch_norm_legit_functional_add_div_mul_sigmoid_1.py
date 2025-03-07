# From: 97_Matmul_BatchNorm_BiasAdd_Divide_Swish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_add_div_mul_sigmoid_1poi_fused__native_batch_norm_legit_functional_add_div_mul_sigmoid_1(
    output_ptr, input_data_ptr, mean_ptr, variance_ptr, scale_ptr, offset_ptr, epsilon_ptr, num_elements, BLOCK_SIZE: tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = block_indices < num_elements
    global_indices = block_indices
    local_indices = block_indices % 512

    data = tl.load(input_data_ptr + (global_indices), mask)
    mean = tl.load(mean_ptr + (local_indices), mask, eviction_policy='evict_last')
    variance = tl.load(variance_ptr + (local_indices), mask, eviction_policy='evict_last')
    scale = tl.load(scale_ptr + (local_indices), mask, eviction_policy='evict_last')
    offset = tl.load(offset_ptr + (local_indices), mask, eviction_policy='evict_last')
    epsilon = tl.load(epsilon_ptr + (0))
    epsilon_broadcast = tl.broadcast_to(epsilon, [BLOCK_SIZE])

    normalized_data = data - mean
    variance_scaled = normalized_data * variance
    scaled_variance = variance_scaled * scale
    shifted_data = scaled_variance + offset
    normalized_output = shifted_data + epsilon_broadcast

    output = normalized_output * tl.sigmoid(normalized_output)
    tl.store(output_ptr + (global_indices), output, mask)