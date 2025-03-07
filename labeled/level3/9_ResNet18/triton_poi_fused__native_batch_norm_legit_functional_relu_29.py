# From: 9_ResNet18

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_29poi_fused__native_batch_norm_legit_functional_relu_29(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_shift, input_ptr_input, 
    output_ptr, num_elements, BLOCK_SIZE: tl.constexpr
):
    num_elements = 50176
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = block_indices < num_elements
    global_indices = block_indices
    local_indices = block_indices % 512

    input_data = tl.load(input_ptr_input + (global_indices), mask)
    mean = tl.load(input_ptr_mean + (local_indices), mask, eviction_policy='evict_last')
    variance = tl.load(input_ptr_var + (local_indices), mask, eviction_policy='evict_last')
    scale = tl.load(input_ptr_scale + (local_indices), mask, eviction_policy='evict_last')
    shift = tl.load(input_ptr_shift + (local_indices), mask, eviction_policy='evict_last')

    centered_data = input_data - mean
    variance_normalized = variance / 98.0
    epsilon = 1e-05
    variance_stabilized = variance_normalized + epsilon
    inv_stddev = tl.extra.cuda.libdevice.rsqrt(variance_stabilized)

    normalized_data = centered_data * inv_stddev
    scaled_data = normalized_data * scale
    shifted_data = scaled_data + shift

    relu_output = tl.full([1], 0, tl.int32)
    relu_result = triton_helpers.maximum(relu_output, shifted_data)

    tl.store(output_ptr + (global_indices), relu_result, mask)