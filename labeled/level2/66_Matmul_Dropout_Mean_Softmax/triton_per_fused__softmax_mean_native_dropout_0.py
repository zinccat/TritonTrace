# From: 66_Matmul_Dropout_Mean_Softmax

import triton
import triton.language as tl


@triton.jit
def triton_per_fused__softmax_mean_native_dropout_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr1, load_seed_offset, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    rnumel = 50
    RBLOCK: tl.constexpr = 64
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_indices = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_indices < rnumel
    row_indices = r_indices
    col_indices = x_indices
    input_data = tl.load(in_ptr1 + (row_indices + (50 * col_indices)), r_mask & x_mask, other=0.0)
    dropout_mask = tl.load(in_ptr2 + (row_indices), r_mask, eviction_policy='evict_last', other=0.0)
    seed_value = tl.load(in_ptr0 + load_seed_offset)
    random_indices = row_indices + (50 * col_indices)
    random_values = tl.rand(seed_value, (random_indices).to(tl.uint32))
    dropout_threshold = 0.2
    dropout_condition = random_values > dropout_threshold
    dropout_mask_float = dropout_condition.to(tl.float32)
    combined_data = input_data + dropout_mask
    scaled_data = dropout_mask_float * combined_data
    scaling_factor = 1.25
    scaled_combined_data = scaled_data * scaling_factor
    broadcasted_data = tl.broadcast_to(scaled_combined_data, [XBLOCK, RBLOCK])
    masked_data = tl.where(r_mask & x_mask, broadcasted_data, 0)
    summed_data = tl.sum(masked_data, 1)[:, None]
    normalization_factor = 50.0
    mean_data = summed_data / normalization_factor
    zero_shifted_data = mean_data - mean_data
    exp_data = tl.math.exp(zero_shifted_data)
    softmax_data = exp_data / exp_data
    tl.store(out_ptr1 + (row_indices + (50 * col_indices)), dropout_condition, r_mask & x_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (col_indices), softmax_data, x_mask)