# From: 27_RegNet

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_max_pool2d_with_indices_mean_25red_fused_max_pool2d_with_indices_mean_25(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    xnumel = 14336
    rnumel = 112
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_col = (x_index % 256)
    x_row = x_index // 256
    _accumulated_max = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x_flat_index = x_index

    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r_flat_index = r_index

        tmp0 = tl.load(in_ptr0 + (x_col + 512 * ((r_flat_index % 28)) + 28672 * (r_flat_index // 28) + 114688 * x_row), r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr0 + (256 + x_col + 512 * ((r_flat_index % 28)) + 28672 * (r_flat_index // 28) + 114688 * x_row), r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr0 + (14336 + x_col + 512 * ((r_flat_index % 28)) + 28672 * (r_flat_index // 28) + 114688 * x_row), r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr0 + (14592 + x_col + 512 * ((r_flat_index % 28)) + 28672 * (r_flat_index // 28) + 114688 * x_row), r_mask & x_mask, eviction_policy='evict_first', other=0.0)

        tmp2 = triton_helpers.maximum(tmp1, tmp0)
        tmp4 = triton_helpers.maximum(tmp3, tmp2)
        tmp6 = triton_helpers.maximum(tmp5, tmp4)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _accumulated_max + tmp7
        _accumulated_max = tl.where(r_mask & x_mask, tmp9, _accumulated_max)

    tmp8 = tl.sum(_accumulated_max, 1)[:, None]
    tl.store(out_ptr0 + (x_flat_index), tmp8, x_mask)