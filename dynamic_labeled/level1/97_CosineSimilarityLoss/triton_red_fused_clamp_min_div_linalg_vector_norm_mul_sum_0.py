# From: 97_CosineSimilarityLoss

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_clamp_min_div_linalg_vector_norm_mul_sum_0(
    output_ptr, input_ptr_a, input_ptr_b, kernel_stride, input_length_a, input_length_b, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < input_length_a
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_indices_flat = x_indices
    squared_sum_a = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for r_offset in range(0, input_length_b, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < input_length_b
        r_indices_flat = r_indices
        loaded_a = tl.load(input_ptr_a + (r_indices_flat + kernel_stride * x_indices_flat), r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        squared_a = loaded_a * loaded_a
        broadcasted_squared_a = tl.broadcast_to(squared_a, [XBLOCK, RBLOCK])
        accumulated_squared_a = squared_sum_a + broadcasted_squared_a
        squared_sum_a = tl.where(r_mask & x_mask, accumulated_squared_a, squared_sum_a)

    sum_squared_a = tl.sum(squared_sum_a, 1)[:, None]
    squared_sum_b = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for r_offset in range(0, input_length_b, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < input_length_b
        r_indices_flat = r_indices
        loaded_b = tl.load(input_ptr_b + (r_indices_flat + kernel_stride * x_indices_flat), r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        squared_b = loaded_b * loaded_b
        broadcasted_squared_b = tl.broadcast_to(squared_b, [XBLOCK, RBLOCK])
        accumulated_squared_b = squared_sum_b + broadcasted_squared_b
        squared_sum_b = tl.where(r_mask & x_mask, accumulated_squared_b, squared_sum_b)

    sum_squared_b = tl.sum(squared_sum_b, 1)[:, None]
    dot_product = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for r_offset in range(0, input_length_b, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < input_length_b
        r_indices_flat = r_indices
        loaded_a = tl.load(input_ptr_a + (r_indices_flat + kernel_stride * x_indices_flat), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        loaded_b = tl.load(input_ptr_b + (r_indices_flat + kernel_stride * x_indices_flat), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        norm_a = tl.extra.cuda.libdevice.sqrt(sum_squared_a)
        epsilon = 1e-08
        clamped_norm_a = triton_helpers.maximum(norm_a, epsilon)
        normalized_a = loaded_a / clamped_norm_a
        norm_b = tl.extra.cuda.libdevice.sqrt(sum_squared_b)
        clamped_norm_b = triton_helpers.maximum(norm_b, epsilon)
        normalized_b = loaded_b / clamped_norm_b
        product = normalized_a * normalized_b
        broadcasted_product = tl.broadcast_to(product, [XBLOCK, RBLOCK])
        accumulated_product = dot_product + broadcasted_product
        dot_product = tl.where(r_mask & x_mask, accumulated_product, dot_product)

    sum_dot_product = tl.sum(dot_product, 1)[:, None]
    tl.store(output_ptr + (x_indices_flat), sum_dot_product, x_mask)