# From: 97_CosineSimilarityLoss

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_red_fused_clamp_min_div_linalg_vector_norm_mul_sum_0(
    output_ptr, input_ptr0, input_ptr1, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    xnumel = 128
    rnumel = 4096
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_indices_0 = x_indices
    sum_squares_0 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    sum_squares_1 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for r_offset in range(0, rnumel, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < rnumel
        r_indices_1 = r_indices
        input_0 = tl.load(input_ptr0 + (r_indices_1 + (4096 * x_indices_0)), r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        input_1 = tl.load(input_ptr1 + (r_indices_1 + (4096 * x_indices_0)), r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        squares_0 = input_0 * input_0
        broadcast_squares_0 = tl.broadcast_to(squares_0, [XBLOCK, RBLOCK])
        updated_sum_squares_0 = sum_squares_0 + broadcast_squares_0
        sum_squares_0 = tl.where(r_mask & x_mask, updated_sum_squares_0, sum_squares_0)
        squares_1 = input_1 * input_1
        broadcast_squares_1 = tl.broadcast_to(squares_1, [XBLOCK, RBLOCK])
        updated_sum_squares_1 = sum_squares_1 + broadcast_squares_1
        sum_squares_1 = tl.where(r_mask & x_mask, updated_sum_squares_1, sum_squares_1)

    sum_squares_0 = tl.sum(sum_squares_0, 1)[:, None]
    sum_squares_1 = tl.sum(sum_squares_1, 1)[:, None]
    product_sum = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for r_offset in range(0, rnumel, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < rnumel
        r_indices_1 = r_indices
        input_0 = tl.load(input_ptr0 + (r_indices_1 + (4096 * x_indices_0)), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        input_1 = tl.load(input_ptr1 + (r_indices_1 + (4096 * x_indices_0)), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        norm_0 = tl.extra.cuda.libdevice.sqrt(sum_squares_0)
        epsilon = 1e-08
        clamped_norm_0 = triton_helpers.maximum(norm_0, epsilon)
        normalized_input_0 = input_0 / clamped_norm_0
        norm_1 = tl.extra.cuda.libdevice.sqrt(sum_squares_1)
        clamped_norm_1 = triton_helpers.maximum(norm_1, epsilon)
        normalized_input_1 = input_1 / clamped_norm_1
        product = normalized_input_0 * normalized_input_1
        broadcast_product = tl.broadcast_to(product, [XBLOCK, RBLOCK])
        updated_product_sum = product_sum + broadcast_product
        product_sum = tl.where(r_mask & x_mask, updated_product_sum, product_sum)

    final_product_sum = tl.sum(product_sum, 1)[:, None]
    tl.store(output_ptr + (x_indices_0), final_product_sum, x_mask)