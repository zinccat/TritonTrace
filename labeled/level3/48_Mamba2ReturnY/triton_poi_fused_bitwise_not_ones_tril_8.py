# From: 48_Mamba2ReturnY

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_bitwise_not_ones_tril_8poi_fused_bitwise_not_ones_tril_8(out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 9
    block_offset = tl.program_id(0) * XBLOCK
    index_within_block = block_offset + tl.arange(0, XBLOCK)[:]
    valid_index_mask = index_within_block < xnumel
    col_index = index_within_block % 3
    row_index = index_within_block // 3
    linear_index = index_within_block
    tmp_col_minus_row = col_index + ((-1) * row_index)
    zero_comparison_value = tl.full([1], 0, tl.int64)
    true_mask = tl.full([1], True, tl.int1)
    is_below_diagonal = tmp_col_minus_row <= zero_comparison_value
    valid_below_diagonal = is_below_diagonal & true_mask
    is_not_below_diagonal = valid_below_diagonal == 0
    tl.store(out_ptr0 + (linear_index), is_not_below_diagonal, valid_index_mask)