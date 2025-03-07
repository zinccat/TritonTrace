# From: 80_Gemm_Max_Subtract_GELU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_div_gelu_backward_neg_scatter_zeros_0poi_fused_add_div_gelu_backward_neg_scatter_zeros_0(
    output_ptr, num_elements, BLOCK_SIZE: tl.constexpr
):
    block_offset = tl.program_id(0) * BLOCK_SIZE
    block_indices = block_offset + tl.arange(0, BLOCK_SIZE)[:]
    valid_mask = block_indices < num_elements
    indices = block_indices
    zero_value = 0.0
    tl.store(output_ptr + (indices), zero_value, valid_mask)