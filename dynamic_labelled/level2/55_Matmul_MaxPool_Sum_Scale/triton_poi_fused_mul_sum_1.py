# From: 55_Matmul_MaxPool_Sum_Scale

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_mul_sum_1poi_fused_mul_sum_1(input_ptr, output_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    indices = block_indices

    input_value_0 = tl.load(input_ptr + (5 * indices), valid_mask, eviction_policy='evict_last')
    input_value_1 = tl.load(input_ptr + (1 + 5 * indices), valid_mask, eviction_policy='evict_last')
    input_value_3 = tl.load(input_ptr + (2 + 5 * indices), valid_mask, eviction_policy='evict_last')
    input_value_4 = tl.load(input_ptr + (3 + 5 * indices), valid_mask, eviction_policy='evict_last')

    max_value_1_0 = triton_helpers.maximum(input_value_1, input_value_0)
    max_value_4_3 = triton_helpers.maximum(input_value_4, input_value_3)

    sum_max_values = max_value_1_0 + max_value_4_3
    scale_factor = 0.5
    scaled_result = sum_max_values * scale_factor

    tl.store(output_ptr + (indices), scaled_result, valid_mask)