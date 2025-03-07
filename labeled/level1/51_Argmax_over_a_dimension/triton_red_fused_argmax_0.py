# From: 51_Argmax_over_a_dimension

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_red_fused_argmax_0(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    rnumel = 256
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_mod_256 = x_indices % 256
    x_div_256 = (x_indices // 256)
    max_values = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    max_indices = tl.full([XBLOCK, RBLOCK], 9223372036854775807, tl.int64)
    x_full_indices = x_indices

    for r_offset in range(0, rnumel, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < rnumel
        r_indices_flat = r_indices
        loaded_values = tl.load(in_ptr0 + (x_mod_256 + (256 * r_indices_flat) + (65536 * x_div_256)), r_mask, eviction_policy='evict_first', other=0.0)
        broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
        max_values_next, max_indices_next = triton_helpers.maximum_with_index(
            max_values, max_indices, broadcasted_values, r_indices
        )
        max_values = tl.where(r_mask, max_values_next, max_values)
        max_indices = tl.where(r_mask, max_indices_next, max_indices)

    _, max_indices_final = triton_helpers.max_with_index(max_values, max_indices, 1)
    max_indices_final = max_indices_final[:, None]
    tl.store(out_ptr0 + (x_full_indices), max_indices_final, None)