# From: 30_SwinTransformerV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__softmax_add_clamp_exp_log_mul_10per_fused__softmax_add_clamp_exp_log_mul_10(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK: tl.constexpr
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
    col_index = x_index
    block_index = x_index // 49
    channel_index = (x_index // 49) % 3
    position_index = x_index % 49

    input_data = tl.load(in_ptr0 + (row_index + 49 * col_index), r_mask & x_mask, other=0.0)
    channel_data = tl.load(in_ptr1 + (channel_index), x_mask, eviction_policy='evict_last')
    reference_data = tl.load(in_ptr2 + (row_index + 49 * position_index), r_mask & x_mask, eviction_policy='evict_last', other=0.0)

    max_log = 4.605170249938965
    clamped_channel_data = triton_helpers.minimum(channel_data, max_log)
    exp_clamped = tl.math.exp(clamped_channel_data)
    weighted_input = input_data * exp_clamped

    bias_value = 169
    bias_tensor = tl.full([XBLOCK, RBLOCK], bias_value, tl.int32)
    biased_reference = reference_data + bias_tensor
    negative_mask = reference_data < 0
    adjusted_reference = tl.where(negative_mask, biased_reference, reference_data)

    tl.device_assert(((0 <= adjusted_reference) & (adjusted_reference < bias_value)) | ~(r_mask & x_mask), "index out of bounds: 0 <= adjusted_reference < 169")

    lookup_index = tl.load(in_ptr3 + (channel_index + 3 * adjusted_reference), r_mask & x_mask, eviction_policy='evict_last')
    sigmoid_output = tl.sigmoid(lookup_index)
    scaled_sigmoid = sigmoid_output * 16.0
    combined_result = weighted_input + scaled_sigmoid

    broadcasted_result = tl.broadcast_to(combined_result, [XBLOCK, RBLOCK])
    max_masked_result = tl.where(r_mask & x_mask, broadcasted_result, float("-inf"))
    max_value = triton_helpers.max2(max_masked_result, 1)[:, None]
    shifted_result = combined_result - max_value
    exp_shifted = tl.math.exp(shifted_result)
    broadcasted_exp = tl.broadcast_to(exp_shifted, [XBLOCK, RBLOCK])
    masked_exp = tl.where(r_mask & x_mask, broadcasted_exp, 0)
    sum_exp = tl.sum(masked_exp, 1)[:, None]
    softmax_result = exp_shifted / sum_exp

    tl.store(in_out_ptr0 + (row_index + 49 * position_index + 2432 * block_index), softmax_result, r_mask & x_mask)