# From: 96_ConvTranspose3d_Multiply_Max_GlobalAvgPool_Clamp

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_mul_1poi_fused_mul_1(in_out_ptr0, num_elements, BLOCK_SIZE: tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    indices = block_indices
    loaded_values = tl.load(in_out_ptr0 + (indices), valid_mask)
    scale_factor = 0.5
    scaled_values = loaded_values * scale_factor
    tl.store(in_out_ptr0 + (indices), scaled_values, valid_mask)