# From: 10_ResNet101

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_add_relu_19poi_fused__native_batch_norm_legit_functional_add_relu_19(
    input_ptr_mean, input_ptr_var, input_ptr_scale, input_ptr_bias, input_ptr_input, output_ptr, num_elements, BLOCK_SIZE: tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    tl.full([BLOCK_SIZE], True, tl.int1)
    global_indices = block_indices
    local_indices = block_indices % 256

    mean = tl.load(input_ptr_mean + (global_indices), None)
    variance = tl.load(input_ptr_var + (local_indices), None, eviction_policy='evict_last')
    scale = tl.load(input_ptr_scale + (local_indices), None, eviction_policy='evict_last')
    bias = tl.load(input_ptr_bias + (local_indices), None, eviction_policy='evict_last')
    input_data = tl.load(input_ptr_input + (global_indices), None)
    epsilon = 1e-05
    rsqrt_denominator = 31360.0

    normalized_input = (input_data - mean) * (tl.extra.cuda.libdevice.rsqrt((variance / rsqrt_denominator) + epsilon))
    scaled_input = normalized_input * scale
    biased_input = scaled_input + bias
    output_data = biased_input + input_data

    zero_tensor = tl.full([1], 0, tl.int32)
    relu_output = triton_helpers.maximum(zero_tensor, output_data)
    tl.store(output_ptr + (global_indices), relu_output, None)