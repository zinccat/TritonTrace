# From: 37_Matmul_Swish_Sum_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_group_norm_1poi_fused_native_group_norm_1(
    input_ptr_mean, input_ptr_var, input_ptr_beta, input_ptr_gamma, input_ptr_input, input_ptr_bias, 
    output_ptr, num_elements, BLOCK_SIZE: tl.constexpr
):
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
    bias = tl.load(input_ptr_bias + (local_indices), mask, eviction_policy='evict_last')

    swish = tl.sigmoid(mean)
    swish_scaled = swish * mean
    normalized = swish_scaled + var
    centered = normalized - beta

    inv_stddev = 32.0
    epsilon = 1e-05
    variance_adjusted = var / inv_stddev
    variance_adjusted_epsilon = variance_adjusted + epsilon
    reciprocal_sqrt = tl.extra.cuda.libdevice.rsqrt(variance_adjusted_epsilon)
    scaled_centered = centered * reciprocal_sqrt
    gamma_scaled = scaled_centered * gamma
    output = gamma_scaled + bias

    tl.store(output_ptr + (global_indices), output, mask)