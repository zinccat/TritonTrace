# From: 55_Matmul_MaxPool_Sum_Scale

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused_mul_sum_1(input_ptr, output_ptr, num_elements, XBLOCK: tl.constexpr):
    num_elements = 128
    offset = tl.program_id(0) * XBLOCK
    indices = offset + tl.arange(0, XBLOCK)[:]
    mask = indices < num_elements
    base_indices = indices

    input_value_0 = tl.load(input_ptr + (5 * base_indices), mask, eviction_policy='evict_last')
    input_value_1 = tl.load(input_ptr + (1 + (5 * base_indices)), mask, eviction_policy='evict_last')
    input_value_2 = tl.load(input_ptr + (2 + (5 * base_indices)), mask, eviction_policy='evict_last')
    input_value_3 = tl.load(input_ptr + (3 + (5 * base_indices)), mask, eviction_policy='evict_last')

    max_value_1 = triton_helpers.maximum(input_value_1, input_value_0)
    max_value_2 = triton_helpers.maximum(input_value_3, input_value_2)

    sum_max_values = max_value_1 + max_value_2
    scale_factor = 0.5
    scaled_result = sum_max_values * scale_factor

    tl.store(output_ptr + (base_indices), scaled_result, mask)