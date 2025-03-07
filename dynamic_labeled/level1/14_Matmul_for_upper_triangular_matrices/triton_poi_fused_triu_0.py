# From: 14_Matmul_for_upper_triangular_matrices

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_triu_0(in_out_ptr0, ks0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    col_index = xindex % ks0
    row_index = xindex // ks0
    linear_index = xindex
    value = tl.load(in_out_ptr0 + (linear_index), xmask, eviction_policy='evict_last')
    upper_tri_index = col_index + ((-1) * row_index)
    zero_value = tl.full([1], 0, tl.int64)
    is_upper_tri = upper_tri_index >= zero_value
    zero_float = 0.0
    result_value = tl.where(is_upper_tri, value, zero_float)
    tl.store(in_out_ptr0 + (linear_index), result_value, xmask)