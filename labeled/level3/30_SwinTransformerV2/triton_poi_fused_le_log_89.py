# From: 30_SwinTransformerV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_le_log_89poi_fused_le_log_89(input_ptr, output_ptr, num_elements, BLOCK_SIZE : tl.constexpr):
    num_elements = 12
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    indices = block_indices
    loaded_values = tl.load(input_ptr + (indices), valid_mask)
    threshold_value = 4.605170249938965
    comparison_result = loaded_values <= threshold_value
    tl.store(output_ptr + (indices), comparison_result, valid_mask)