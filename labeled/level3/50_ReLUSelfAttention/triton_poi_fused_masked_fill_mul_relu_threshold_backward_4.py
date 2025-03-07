# From: 50_ReLUSelfAttention

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_masked_fill_mul_relu_threshold_backward_4poi_fused_masked_fill_mul_relu_threshold_backward_4(
    in_out_ptr, input_ptr, output_ptr, num_elements, BLOCK_SIZE: tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    mask = tl.full([BLOCK_SIZE], True, tl.int1)
    index_modulo = block_indices % 1048576
    linear_index = block_indices
    input_mask = tl.load(input_ptr + (index_modulo), None, eviction_policy='evict_last').to(tl.int1)
    input_output_values = tl.load(in_out_ptr + (linear_index), None)
    scale_factor = 0.125
    scaled_values = input_output_values * scale_factor
    negative_infinity = float("-inf")
    masked_values = tl.where(input_mask, negative_infinity, scaled_values)
    zero_tensor = tl.full([1], 0, tl.int32)
    max_values = triton_helpers.maximum(zero_tensor, masked_values)
    zero_float = 0.0
    threshold_mask = max_values <= zero_float
    tl.store(in_out_ptr + (linear_index), max_values, None)
    tl.store(output_ptr + (linear_index), threshold_mask, None)