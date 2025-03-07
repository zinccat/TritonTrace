# From: 9_ResNet18

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_add_relu_27poi_fused__native_batch_norm_legit_functional_add_relu_27(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_bias, input_ptr_add, output_ptr, num_elements, BLOCK_SIZE: tl.constexpr
):
    num_elements = 100352
    block_offset = tl.program_id(0) * BLOCK_SIZE
    element_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = element_indices < num_elements
    global_indices = element_indices
    channel_indices = element_indices % 256

    mean = tl.load(input_ptr_mean + (global_indices), valid_mask)
    variance = tl.load(input_ptr_var + (channel_indices), valid_mask, eviction_policy='evict_last')
    scale = tl.load(input_ptr_scale + (channel_indices), valid_mask, eviction_policy='evict_last')
    bias = tl.load(input_ptr_bias + (channel_indices), valid_mask, eviction_policy='evict_last')
    add = tl.load(input_ptr_add + (global_indices), valid_mask)
    
    normalized = mean - variance
    variance_normalized = variance / 392.0
    epsilon = 1e-05
    variance_stabilized = variance_normalized + epsilon
    inv_stddev = tl.extra.cuda.libdevice.rsqrt(variance_stabilized)
    scaled_normalized = normalized * inv_stddev
    scaled = scaled_normalized * scale
    biased = scaled + bias
    result = biased + add

    zero_tensor = tl.full([1], 0, tl.int32)
    relu_result = triton_helpers.maximum(zero_tensor, result)
    tl.store(output_ptr + (global_indices), relu_result, valid_mask)