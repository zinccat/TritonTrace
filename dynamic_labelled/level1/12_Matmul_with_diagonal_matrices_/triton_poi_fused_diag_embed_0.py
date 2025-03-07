# From: 12_Matmul_with_diagonal_matrices_

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_diag_embed_0(in_ptr0, out_ptr0, ks0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    col_index = xindex % ks0
    row_index = xindex // ks0
    linear_index = xindex
    input_value = tl.load(in_ptr0 + (col_index), xmask, eviction_policy='evict_last')
    col = col_index
    row = row_index
    is_diagonal = col == row
    default_value = 0.0
    output_value = tl.where(is_diagonal, input_value, default_value)
    tl.store(out_ptr0 + (linear_index), output_value, xmask)