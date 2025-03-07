# From: 37_Matmul_Swish_Sum_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_group_norm_1(input_ptr_mean, input_ptr_var, input_ptr_beta, input_ptr_gamma, input_ptr_input, input_ptr_output, output_ptr, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = block_indices < num_elements
    global_indices = block_indices
    local_indices = block_indices % 1024

    mean = tl.load(input_ptr_mean + (global_indices), mask)
    var = tl.load(input_ptr_var + (local_indices), mask, eviction_policy='evict_last')
    beta = tl.load(input_ptr_beta + (global_indices // 32), mask, eviction_policy='evict_last')
    gamma = tl.load(input_ptr_gamma + (global_indices // 32), mask, eviction_policy='evict_last')
    input_data = tl.load(input_ptr_input + (local_indices), mask, eviction_policy='evict_last')
    output_data = tl.load(input_ptr_output + (local_indices), mask, eviction_policy='evict_last')

    swish_activation = tl.sigmoid(mean)
    swish_result = swish_activation * mean
    normalized_input = swish_result + var
    centered_input = normalized_input - beta

    inv_stddev = 32.0
    epsilon = 1e-05
    adjusted_var = var / inv_stddev
    variance_with_epsilon = adjusted_var + epsilon
    reciprocal_sqrt = tl.extra.cuda.libdevice.rsqrt(variance_with_epsilon)

    scaled_input = centered_input * reciprocal_sqrt
    scaled_output = scaled_input * gamma
    final_output = scaled_output + output_data

    tl.store(output_ptr + (global_indices), final_output, mask)