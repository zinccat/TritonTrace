# From: 15_Matmul_for_lower_triangular_matrices

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_tril_0poi_fused_tril_0(in_out_ptr0, kernel_size, num_elements, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_indices = block_offset + tl.arange(0, XBLOCK)[:]
    valid_mask = block_indices < num_elements
    col_index = block_indices % kernel_size
    row_index = block_indices // kernel_size
    linear_index = block_indices
    loaded_value = tl.load(in_out_ptr0 + (linear_index), valid_mask, eviction_policy='evict_last')
    lower_triangular_condition = col_index + ((-1) * row_index)
    zero_value = tl.full([1], 0, tl.int64)
    is_lower_triangular = lower_triangular_condition <= zero_value
    zero_float = 0.0
    result_value = tl.where(is_lower_triangular, loaded_value, zero_float)
    tl.store(in_out_ptr0 + (linear_index), result_value, valid_mask)