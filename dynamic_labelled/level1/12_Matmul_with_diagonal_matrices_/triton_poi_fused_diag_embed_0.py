# From: 12_Matmul_with_diagonal_matrices_

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_diag_embed_0poi_fused_diag_embed_0(in_ptr0, out_ptr0, kernel_size, num_elements, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    valid_mask = block_indices < num_elements
    
    col_index = block_indices % kernel_size
    row_index = block_indices // kernel_size
    linear_index = block_indices
    
    input_value = tl.load(in_ptr0 + (col_index), valid_mask, eviction_policy='evict_last')
    col = col_index
    row = row_index
    is_diagonal = col == row
    
    zero_value = 0.0
    output_value = tl.where(is_diagonal, input_value, zero_value)
    
    tl.store(out_ptr0 + (linear_index), output_value, valid_mask)