# From: 72_ConvTranspose3d_BatchNorm_AvgPool_AvgPool

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_1(input_ptr, output_mean_ptr, output_variance_ptr, output_count_ptr, total_elements, reduction_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    total_elements = 3984
    reduction_elements = 128539
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < total_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_dim1 = x_indices % 249
    x_dim2 = (x_indices // 249)
    mean_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    m2_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    count_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x_linear_index = x_indices

    for r_offset in range(0, reduction_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < reduction_elements
        r_linear_index = r_indices
        combined_index = r_linear_index + (128539 * x_dim1)
        max_index = tl.full([1, 1], 32006016, tl.int32)
        index_within_bounds = combined_index < max_index
        loaded_values = tl.load(input_ptr + ((250047 * x_dim2) + (4000752 * (((r_linear_index + (128539 * x_dim1)) // 250047) % 128)) + ((r_linear_index + (128539 * x_dim1)) % 250047)), r_mask & index_within_bounds & x_mask, eviction_policy='evict_last', other=0.0)
        zero_tensor = tl.full(loaded_values.shape, 0, loaded_values.dtype)
        mask_tensor = tl.where(index_within_bounds, 0.0, zero_tensor)
        one_tensor = tl.full(loaded_values.shape, 0, 1.0)
        count_tensor = tl.where(index_within_bounds, 1.0, one_tensor)
        broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
        broadcasted_mask = tl.broadcast_to(mask_tensor, [XBLOCK, RBLOCK])
        broadcasted_count = tl.broadcast_to(count_tensor, [XBLOCK, RBLOCK])
        mean_next, m2_next, count_next = triton_helpers.welford_combine(
            mean_accumulator, m2_accumulator, count_accumulator,
            broadcasted_values, broadcasted_mask, broadcasted_count
        )
        mean_accumulator = tl.where(r_mask & x_mask, mean_next, mean_accumulator)
        m2_accumulator = tl.where(r_mask & x_mask, m2_next, m2_accumulator)
        count_accumulator = tl.where(r_mask & x_mask, count_next, count_accumulator)

    mean_result, variance_result, count_result = triton_helpers.welford(
        mean_accumulator, m2_accumulator, count_accumulator, 1
    )
    mean_result = mean_result[:, None]
    variance_result = variance_result[:, None]
    count_result = count_result[:, None]
    tl.store(output_mean_ptr + (x_linear_index), mean_result, x_mask)
    tl.store(output_variance_ptr + (x_linear_index), variance_result, x_mask)
    tl.store(output_count_ptr + (x_linear_index), count_result, x_mask)