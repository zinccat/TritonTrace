# From: 97_Matmul_BatchNorm_BiasAdd_Divide_Swish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_add_div_0(
    input_data_ptr, mean_ptr, variance_ptr, scale_ptr, offset_ptr, bias_ptr, 
    output_ptr, num_elements, BLOCK_SIZE : tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    global_indices = block_indices
    local_indices = block_indices % 512

    input_data = tl.load(input_data_ptr + (global_indices), valid_mask)
    mean = tl.load(mean_ptr + (local_indices), valid_mask, eviction_policy='evict_last')
    variance = tl.load(variance_ptr + (local_indices), valid_mask, eviction_policy='evict_last')
    scale = tl.load(scale_ptr + (local_indices), valid_mask, eviction_policy='evict_last')
    offset = tl.load(offset_ptr + (local_indices), valid_mask, eviction_policy='evict_last')
    bias = tl.load(bias_ptr + (0))
    broadcast_bias = tl.broadcast_to(bias, [BLOCK_SIZE])

    normalized_data = input_data - mean
    scaled_variance = normalized_data * variance
    scaled_data = scaled_variance * scale
    shifted_data = scaled_data + offset
    result = shifted_data + broadcast_bias

    final_output = result * 1.0
    tl.store(output_ptr + (global_indices), final_output, valid_mask)