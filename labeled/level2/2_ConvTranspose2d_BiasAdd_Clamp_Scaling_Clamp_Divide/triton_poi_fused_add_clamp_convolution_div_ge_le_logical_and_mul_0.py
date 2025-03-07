# From: 2_ConvTranspose2d_BiasAdd_Clamp_Scaling_Clamp_Divide

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused_add_clamp_convolution_div_ge_le_logical_and_mul_0(
    input_ptr0, input_ptr1, input_ptr2, output_ptr0, output_ptr1, output_ptr2, num_elements, XBLOCK: tl.constexpr
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    block_index = index
    block_id = (index // 4096) % 16

    input_value0 = tl.load(input_ptr0 + (block_index), None)
    input_value1 = tl.load(input_ptr1 + (block_id), None, eviction_policy='evict_last')
    input_value2 = tl.load(input_ptr2 + (block_id), None, eviction_policy='evict_last')

    sum_values = input_value0 + input_value1
    total_sum = sum_values + input_value2

    clamp_min = 0.0
    max_value = triton_helpers.maximum(total_sum, clamp_min)

    clamp_max = 1.0
    clamped_value = triton_helpers.minimum(max_value, clamp_max)

    scale_factor = 2.0
    scaled_value = clamped_value * scale_factor

    max_scaled = triton_helpers.maximum(scaled_value, clamp_min)
    final_clamped = triton_helpers.minimum(max_scaled, clamp_max)

    scale_divisor = 0.5
    result_value = final_clamped * scale_divisor

    scaled_ge_min = scaled_value >= clamp_min
    scaled_le_max = scaled_value <= clamp_max
    scaled_in_range = scaled_ge_min & scaled_le_max

    original_ge_min = total_sum >= clamp_min
    original_le_max = total_sum <= clamp_max
    original_in_range = original_ge_min & original_le_max

    tl.store(output_ptr0 + (block_index), result_value, None)
    tl.store(output_ptr1 + (block_index), scaled_in_range, None)
    tl.store(output_ptr2 + (block_index), original_in_range, None)