# From: 33_Gemm_Scale_BatchNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_mul_1(input_ptr, scale_ptr, bias_ptr, mean_ptr, var_ptr, epsilon_ptr, output_ptr, scale_factor, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = block_indices < num_elements
    global_indices = block_indices
    local_indices = block_indices % 512

    input_data = tl.load(input_ptr + (global_indices), mask)
    scale_data = tl.load(scale_ptr + (local_indices), mask, eviction_policy='evict_last')
    bias_data = tl.load(bias_ptr + (local_indices), mask, eviction_policy='evict_last')
    mean_data = tl.load(mean_ptr + (local_indices), mask, eviction_policy='evict_last')
    var_data = tl.load(var_ptr + (local_indices), mask, eviction_policy='evict_last')
    epsilon_data = tl.load(epsilon_ptr + (local_indices), mask, eviction_policy='evict_last')

    scaled_input = input_data * scale_data
    centered_input = scaled_input - mean_data
    scale_factor_float = scale_factor.to(tl.float32)
    inv_std_dev = var_data / scale_factor_float
    epsilon_value = 1e-05
    adjusted_var = inv_std_dev + epsilon_value
    reciprocal_sqrt = tl.extra.cuda.libdevice.rsqrt(adjusted_var)
    normalized_input = centered_input * reciprocal_sqrt
    scaled_normalized_input = normalized_input * scale_data
    output_data = scaled_normalized_input + bias_data

    tl.store(output_ptr + (global_indices), output_data, mask)