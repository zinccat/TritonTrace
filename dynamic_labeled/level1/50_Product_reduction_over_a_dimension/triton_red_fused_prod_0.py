# From: 50_Product_reduction_over_a_dimension

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_prod_0(in_ptr0, out_ptr0, ks0, ks1, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x1 = (x_index // ks0) % 2
    x0 = x_index % ks0
    x2 = x_index // ks1
    initial_product = tl.full([XBLOCK, RBLOCK], 1, tl.float32)
    x4 = x_index

    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r3 = r_index
        adjusted_index = r3 + x1 * ((1 + ks0) // 2)
        ks0_limit = ks0
        valid_mask = adjusted_index < ks0_limit
        load_mask = r_mask & valid_mask & x_mask
        loaded_values = tl.load(in_ptr0 + (x0 + ks0 * r3 + x2 * ks0 * ks0 + ks0 * x1 * ((1 + ks0) // 2)), load_mask, eviction_policy='evict_last', other=1)
        broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
        updated_product = initial_product * broadcasted_values
        initial_product = tl.where(r_mask & x_mask, updated_product, initial_product)

    final_product = triton_helpers.prod(initial_product, 1)[:, None]
    tl.store(out_ptr0 + (x4), final_product, x_mask)