# From: 30_SwinTransformerV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clamp_exp_log_78poi_fused_clamp_exp_log_78(input_ptr, output_ptr, num_elements, BLOCK_SIZE : tl.constexpr):
    num_elements = 24
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    indices = block_indices
    input_values = tl.load(input_ptr + (indices), valid_mask)
    clamp_value = 4.605170249938965
    clamped_values = triton_helpers.minimum(input_values, clamp_value)
    exp_values = tl.math.exp(clamped_values)
    tl.store(output_ptr + (indices), exp_values, valid_mask)