# From: 97_Matmul_BatchNorm_BiasAdd_Divide_Swish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_add_div_0poi_fused__native_batch_norm_legit_functional_add_div_0(
    input_mean_ptr, input_var_ptr, input_scale_ptr, input_offset_ptr, input_bias_ptr, input_gamma_ptr, 
    output_ptr, num_elements, BLOCK_SIZE : tl.constexpr):

    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    global_indices = block_indices
    local_indices = block_indices % 512

    mean = tl.load(input_mean_ptr + (global_indices), valid_mask)
    variance = tl.load(input_var_ptr + (local_indices), valid_mask, eviction_policy='evict_last')
    scale = tl.load(input_scale_ptr + (local_indices), valid_mask, eviction_policy='evict_last')
    offset = tl.load(input_offset_ptr + (local_indices), valid_mask, eviction_policy='evict_last')
    bias = tl.load(input_bias_ptr + (local_indices), valid_mask, eviction_policy='evict_last')
    gamma = tl.load(input_gamma_ptr + (0))
    broadcast_gamma = tl.broadcast_to(gamma, [BLOCK_SIZE])

    normalized = mean - variance
    scaled = normalized * scale
    scaled_var = scaled * variance
    biased = scaled_var + offset
    biased_with_gamma = biased + broadcast_gamma

    result = biased_with_gamma * 1.0
    tl.store(output_ptr + (global_indices), result, valid_mask)