# From: 50_Product_reduction_over_a_dimension

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_red_fused_prod_0(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    rnumel = 128
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_mod_256 = x_indices % 256
    x_div_256 = (x_indices // 256)
    initial_product = tl.full([XBLOCK, RBLOCK], 1, tl.float32)
    x_full_indices = x_indices

    for r_offset in range(0, rnumel, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < rnumel
        r_indices_flat = r_indices
        loaded_values = tl.load(in_ptr0 + (x_mod_256 + (256 * r_indices_flat) + (32768 * x_div_256)), r_mask, eviction_policy='evict_first', other=0.0)
        broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
        updated_product = initial_product * broadcasted_values
        initial_product = tl.where(r_mask, updated_product, initial_product)

    final_product = triton_helpers.prod(initial_product, 1)[:, None]
    tl.store(out_ptr0 + (x_full_indices), final_product, None)