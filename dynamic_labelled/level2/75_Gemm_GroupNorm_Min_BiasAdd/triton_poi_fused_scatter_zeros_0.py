# From: 75_Gemm_GroupNorm_Min_BiasAdd

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_scatter_zeros_0(output_ptr, num_elements, BLOCK_SIZE : tl.constexpr):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    element_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = element_indices < num_elements
    indices = element_indices
    zero_value = 0.0
    tl.store(output_ptr + (indices), zero_value, valid_mask)