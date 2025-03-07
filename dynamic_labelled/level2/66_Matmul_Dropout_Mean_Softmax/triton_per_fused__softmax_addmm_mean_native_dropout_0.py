# From: 66_Matmul_Dropout_Mean_Softmax

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused__softmax_addmm_mean_native_dropout_0(
    in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr1, load_seed_offset, xnumel, rnumel, XBLOCK: tl.constexpr
):
    rnumel = 50
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    x_indices = xoffset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < xnumel
    r_indices = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_indices < rnumel
    row_indices = r_indices
    col_indices = x_indices
    input_data1 = tl.load(in_ptr1 + (row_indices + 50 * col_indices), r_mask & x_mask, other=0.0)
    input_data2 = tl.load(in_ptr2 + (row_indices), r_mask, eviction_policy='evict_last', other=0.0)
    random_seed = tl.load(in_ptr0 + load_seed_offset)
    index_for_random = row_indices + 50 * col_indices
    random_values = tl.rand(random_seed, (index_for_random).to(tl.uint32))
    dropout_threshold = 0.2
    dropout_mask = random_values > dropout_threshold
    dropout_mask_float = dropout_mask.to(tl.float32)
    combined_data = input_data1 + input_data2
    dropout_applied = dropout_mask_float * combined_data
    dropout_scale = 1.25
    scaled_dropout = dropout_applied * dropout_scale
    broadcasted_dropout = tl.broadcast_to(scaled_dropout, [XBLOCK, RBLOCK])
    masked_dropout = tl.where(r_mask & x_mask, broadcasted_dropout, 0)
    sum_masked_dropout = tl.sum(masked_dropout, 1)[:, None]
    normalization_factor = 50.0
    mean_values = sum_masked_dropout / normalization_factor
    zeroed_mean = mean_values - mean_values
    exp_values = tl.math.exp(zeroed_mean)
    softmax_output = exp_values / exp_values
    tl.store(out_ptr1 + (row_indices + 50 * col_indices), dropout_mask, r_mask & x_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (col_indices), softmax_output, x_mask)