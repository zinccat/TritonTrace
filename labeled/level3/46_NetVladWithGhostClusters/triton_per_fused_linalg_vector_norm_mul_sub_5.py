# From: 46_NetVladWithGhostClusters

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_linalg_vector_norm_mul_sub_5(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel
):
    XBLOCK: tl.constexpr = 1
    RBLOCK: tl.constexpr = 512

    x_offset = tl.program_id(0) * XBLOCK
    x_index = tl.full([1], x_offset, tl.int32)
    r_mask = tl.full([RBLOCK], True, tl.int1)
    r_index = tl.arange(0, RBLOCK)[:]
    r_mask = tl.full([RBLOCK], True, tl.int1)
    r2 = r_index
    x3 = x_index
    x0 = (x_index % 32)

    tmp0 = tl.load(in_ptr0 + (r2 + 512 * x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x0 + 32 * r2), None, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 - tmp3
    tmp5 = tmp4 * tmp4
    tmp6 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp9 = tl.extra.cuda.libdevice.sqrt(tmp8)

    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp9, None)