# From: 97_CosineSimilarityLoss

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_clamp_min_div_linalg_vector_norm_mul_sum_0(
    output_ptr, input_ptr0, input_ptr1, kernel_size, input_num_elements, reduction_num_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < input_num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_indices_0 = x_indices

    squared_sum_0 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for r_offset in range(0, reduction_num_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < reduction_num_elements
        r_indices_1 = r_indices
        loaded_values_0 = tl.load(input_ptr0 + (r_indices_1 + kernel_size * x_indices_0), r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        squared_values_0 = loaded_values_0 * loaded_values_0
        broadcasted_squared_0 = tl.broadcast_to(squared_values_0, [XBLOCK, RBLOCK])
        accumulated_squared_0 = squared_sum_0 + broadcasted_squared_0
        squared_sum_0 = tl.where(r_mask & x_mask, accumulated_squared_0, squared_sum_0)

    sum_squared_0 = tl.sum(squared_sum_0, 1)[:, None]

    squared_sum_1 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for r_offset in range(0, reduction_num_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < reduction_num_elements
        r_indices_1 = r_indices
        loaded_values_1 = tl.load(input_ptr1 + (r_indices_1 + kernel_size * x_indices_0), r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        squared_values_1 = loaded_values_1 * loaded_values_1
        broadcasted_squared_1 = tl.broadcast_to(squared_values_1, [XBLOCK, RBLOCK])
        accumulated_squared_1 = squared_sum_1 + broadcasted_squared_1
        squared_sum_1 = tl.where(r_mask & x_mask, accumulated_squared_1, squared_sum_1)

    sum_squared_1 = tl.sum(squared_sum_1, 1)[:, None]

    cosine_similarity = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for r_offset in range(0, reduction_num_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < reduction_num_elements
        r_indices_1 = r_indices
        loaded_values_0 = tl.load(input_ptr0 + (r_indices_1 + kernel_size * x_indices_0), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        loaded_values_1 = tl.load(input_ptr1 + (r_indices_1 + kernel_size * x_indices_0), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        norm_0 = tl.extra.cuda.libdevice.sqrt(sum_squared_0)
        epsilon = 1e-08
        clamped_norm_0 = triton_helpers.maximum(norm_0, epsilon)
        normalized_values_0 = loaded_values_0 / clamped_norm_0
        norm_1 = tl.extra.cuda.libdevice.sqrt(sum_squared_1)
        clamped_norm_1 = triton_helpers.maximum(norm_1, epsilon)
        normalized_values_1 = loaded_values_1 / clamped_norm_1
        product = normalized_values_0 * normalized_values_1
        broadcasted_product = tl.broadcast_to(product, [XBLOCK, RBLOCK])
        accumulated_product = cosine_similarity + broadcasted_product
        cosine_similarity = tl.where(r_mask & x_mask, accumulated_product, cosine_similarity)

    sum_cosine_similarity = tl.sum(cosine_similarity, 1)[:, None]
    tl.store(output_ptr + (x_indices_0), sum_cosine_similarity, x_mask)