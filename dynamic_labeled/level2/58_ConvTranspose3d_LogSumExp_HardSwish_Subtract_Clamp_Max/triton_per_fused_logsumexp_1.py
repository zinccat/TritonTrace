# From: 58_ConvTranspose3d_LogSumExp_HardSwish_Subtract_Clamp_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_logsumexp_1(in_ptr0, out_ptr0, out_ptr1, ks0, ks1, ks2, xnumel, rnumel, XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 16
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_index = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = r_index
    x3 = (x_index % ks0)
    x4 = x_index // ks0
    x5 = x_index

    # Load input with complex indexing
    tmp0 = tl.load(
        in_ptr0 + (
            x3 + ((-1) * r2) + ((-16) * x4) + ((-64) * x4 * ks2 * ks2) +
            ((-4) * r2 * ks2 * ks2) + 2 * ks1 * r2 + 4 * ks2 * r2 +
            32 * ks1 * x4 + 64 * ks2 * x4 + ((-128) * ks1 * ks2 * x4) +
            ((-8) * ks1 * ks2 * r2) + 8 * ks1 * r2 * ks2 * ks2 +
            128 * ks1 * x4 * ks2 * ks2
        ),
        x_mask,
        eviction_policy='evict_last',
        other=0.0
    )

    # Broadcast and apply mask
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(x_mask, tmp1, float("-inf"))

    # Compute max and handle infinities
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tmp5 = tl.math.abs(tmp4)
    tmp6 = float("inf")
    tmp7 = tmp5 == tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp7, tmp8, tmp4)

    # Compute exponentials and sum
    tmp10 = tmp0 - tmp9
    tmp11 = tl.math.exp(tmp10)
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
    tmp14 = tl.where(x_mask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]

    # Store results
    tl.store(out_ptr0 + (x5), tmp4, x_mask)
    tl.store(out_ptr1 + (x5), tmp15, x_mask)