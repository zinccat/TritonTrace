# From: 46_NetVladWithGhostClusters

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_linalg_vector_norm_7per_fused_linalg_vector_norm_7(input_ptr, output_ptr, num_elements, reduction_num_elements, XBLOCK: tl.constexpr):
    num_elements = 64
    RBLOCK: tl.constexpr = 64
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements
    r_indices = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = r_indices
    x0 = x_indices
    tmp0 = tl.load(input_ptr + (r1 + 64 * x0), x_mask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(x_mask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(output_ptr + (x0), tmp4, x_mask)