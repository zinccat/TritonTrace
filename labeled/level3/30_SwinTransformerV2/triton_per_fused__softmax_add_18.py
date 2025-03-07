# From: 30_SwinTransformerV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__softmax_add_18per_fused__softmax_add_18(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel, XBLOCK: tl.constexpr
):
    xnumel = 94080
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < xnumel
    r_index = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_index < rnumel
    row_index = r_index
    x_index_flat = x_index
    channel_index = (x_index // 49) % 3
    spatial_index = x_index % 49
    depth_index = (x_index // 147) % 64
    batch_index = x_index // 49

    tmp_input0 = tl.load(in_ptr0 + (row_index + 49 * x_index_flat), r_mask & x_mask, other=0.0)
    tmp_channel_weights = tl.load(in_ptr1 + (channel_index), x_mask, eviction_policy='evict_last')
    tmp_input1 = tl.load(in_ptr2 + (row_index + 49 * spatial_index), r_mask & x_mask, eviction_policy='evict_last', other=0.0)
    tmp_input4 = tl.load(in_ptr4 + (row_index + 49 * spatial_index + 2401 * depth_index), r_mask & x_mask, eviction_policy='evict_last', other=0.0)

    tmp_weighted_input = tmp_input0 * tmp_channel_weights
    tmp_bias = tl.full([XBLOCK, RBLOCK], 169, tl.int32)
    tmp_bias_added = tmp_input1 + tmp_bias
    tmp_negative_mask = tmp_input1 < 0
    tmp_clipped_input = tl.where(tmp_negative_mask, tmp_bias_added, tmp_input1)

    tl.device_assert(((0 <= tmp_clipped_input) & (tmp_clipped_input < 169)) | ~(r_mask & x_mask), "index out of bounds: 0 <= tmp_clipped_input < 169")

    tmp_weights = tl.load(in_ptr3 + (channel_index + 3 * tmp_clipped_input), r_mask & x_mask, eviction_policy='evict_last')
    tmp_sigmoid = tl.sigmoid(tmp_weights)
    tmp_scaled_sigmoid = tmp_sigmoid * 16.0
    tmp_combined_input = tmp_weighted_input + tmp_scaled_sigmoid
    tmp_final_input = tmp_combined_input + tmp_input4

    tmp_broadcasted_input = tl.broadcast_to(tmp_final_input, [XBLOCK, RBLOCK])
    tmp_softmax_input = tl.where(r_mask & x_mask, tmp_broadcasted_input, float("-inf"))
    tmp_max_value = triton_helpers.max2(tmp_softmax_input, 1)[:, None]
    tmp_normalized_input = tmp_final_input - tmp_max_value
    tmp_exp_input = tl.math.exp(tmp_normalized_input)
    tmp_broadcasted_exp = tl.broadcast_to(tmp_exp_input, [XBLOCK, RBLOCK])
    tmp_masked_exp = tl.where(r_mask & x_mask, tmp_broadcasted_exp, 0)
    tmp_sum_exp = tl.sum(tmp_masked_exp, 1)[:, None]
    tmp_softmax_output = tmp_exp_input / tmp_sum_exp

    tl.store(in_out_ptr0 + (row_index + 49 * spatial_index + 2432 * batch_index), tmp_softmax_output, r_mask & x_mask)