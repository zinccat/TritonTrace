# From: 37_FrobeniusNorm_

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_per_fused_linalg_vector_norm_1(in_ptr0, out_ptr0, xnumel, rnumel):
    XBLOCK: tl.constexpr = 1
    rnumel = 328
    RBLOCK: tl.constexpr = 512
    x_offset = tl.program_id(0) * XBLOCK
    tl.full([1], x_offset, tl.int32)
    tl.full([RBLOCK], True, tl.int1)
    r_index = tl.arange(0, RBLOCK)[:]
    r_mask = r_index < rnumel
    r_indices = r_index
    loaded_values = tl.load(in_ptr0 + (r_indices), r_mask, other=0.0)
    broadcasted_values = tl.broadcast_to(loaded_values, [RBLOCK])
    masked_values = tl.where(r_mask, broadcasted_values, 0)
    sum_result = triton_helpers.promote_to_tensor(tl.sum(masked_values, 0))
    tl.store(out_ptr0 + (tl.full([1], 0, tl.int32)), sum_result, None)