# From: 20_MobileNetV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_hardtanh_mean_64(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel, XBLOCK: tl.constexpr
):
    xnumel = 12800
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_indices = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_indices < rnumel
    r2 = r_indices
    x0 = (x_indices % 1280)
    x1 = x_indices // 1280
    x3 = x_indices

    tmp0 = tl.load(in_ptr0 + (x0 + 1280 * r2 + 62720 * x1), r_mask & x_mask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), x_mask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), x_mask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0), x_mask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x0), x_mask, eviction_policy='evict_last')

    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7

    tmp9 = 0.0
    tmp10 = triton_helpers.maximum(tmp8, tmp9)
    tmp11 = 6.0
    tmp12 = triton_helpers.minimum(tmp10, tmp11)

    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(r_mask & x_mask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = 49.0
    tmp18 = tmp16 / tmp17

    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp18, x_mask)