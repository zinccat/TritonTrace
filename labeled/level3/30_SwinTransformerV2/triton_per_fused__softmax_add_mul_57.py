# From: 30_SwinTransformerV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__softmax_add_mul_57per_fused__softmax_add_mul_57(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK: tl.constexpr
):
    xnumel = 23520
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_index = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_index < rnumel
    row_index = r_index
    col_index = x_index
    block_row_index = x_index // 49
    col_within_block = x_index % 49
    block_index = x_index // 49

    # Load data with masking
    input_data_0 = tl.load(in_ptr0 + (row_index + 49 * col_index), r_mask & x_mask, other=0.0)
    input_data_1 = tl.load(in_ptr1 + (block_row_index % 12), x_mask, eviction_policy='evict_last')
    input_data_2 = tl.load(in_ptr2 + (row_index + 49 * col_within_block), r_mask & x_mask, eviction_policy='evict_last', other=0.0)

    # Perform operations
    multiplied_data = input_data_0 * input_data_1
    constant_tensor = tl.full([XBLOCK, RBLOCK], 169, tl.int32)
    added_data = input_data_2 + constant_tensor
    negative_mask = added_data < 0
    adjusted_data = tl.where(negative_mask, added_data, added_data)

    # Assert index bounds
    tl.device_assert(((0 <= adjusted_data) & (adjusted_data < 169)) | ~(r_mask & x_mask), "index out of bounds: 0 <= adjusted_data < 169")

    # Load and apply sigmoid
    loaded_data = tl.load(in_ptr3 + (block_row_index % 12 + 12 * adjusted_data), r_mask & x_mask, eviction_policy='evict_last')
    sigmoid_applied = tl.sigmoid(loaded_data)
    scaled_sigmoid = sigmoid_applied * 16.0

    # Combine results
    combined_data = multiplied_data + scaled_sigmoid
    broadcasted_data = tl.broadcast_to(combined_data, [XBLOCK, RBLOCK])
    masked_data = tl.where(r_mask & x_mask, broadcasted_data, float("-inf"))

    # Softmax computation
    max_values = triton_helpers.max2(masked_data, 1)[:, None]
    shifted_data = combined_data - max_values
    exp_data = tl.math.exp(shifted_data)
    broadcasted_exp = tl.broadcast_to(exp_data, [XBLOCK, RBLOCK])
    masked_exp = tl.where(r_mask & x_mask, broadcasted_exp, 0)
    sum_exp = tl.sum(masked_exp, 1)[:, None]
    softmax_result = exp_data / sum_exp

    # Store result
    tl.store(in_out_ptr0 + (row_index + 49 * col_within_block + 2432 * block_index), softmax_result, r_mask & x_mask)