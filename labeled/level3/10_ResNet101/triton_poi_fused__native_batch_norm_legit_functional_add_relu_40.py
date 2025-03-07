# From: 10_ResNet101

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_add_relu_40poi_fused__native_batch_norm_legit_functional_add_relu_40(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_bias, input_ptr_add, output_ptr, num_elements, BLOCK_SIZE: tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    global_indices = block_indices
    local_indices = block_indices % 1024

    mean = tl.load(input_ptr_mean + (global_indices), None)
    variance = tl.load(input_ptr_var + (local_indices), None, eviction_policy='evict_last')
    scale = tl.load(input_ptr_scale + (local_indices), None, eviction_policy='evict_last')
    bias = tl.load(input_ptr_bias + (local_indices), None, eviction_policy='evict_last')
    add = tl.load(input_ptr_add + (global_indices), None)

    normalized = mean - variance
    variance_normalized = 1960.0
    variance_adjusted = variance / variance_normalized
    epsilon = 1e-05
    variance_stabilized = variance_adjusted + epsilon
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_stabilized)
    scaled_normalized = normalized * inv_sqrt_variance
    scaled = scaled_normalized * scale
    biased = scaled + bias
    result = biased + add

    zero = tl.full([1], 0, tl.int32)
    relu_result = triton_helpers.maximum(zero, result)
    tl.store(output_ptr + (global_indices), relu_result, None)