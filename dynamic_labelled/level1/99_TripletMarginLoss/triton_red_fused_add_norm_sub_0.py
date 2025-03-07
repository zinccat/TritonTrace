# From: 99_TripletMarginLoss

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_add_norm_sub_0red_fused_add_norm_sub_0(
    input_ptr0, input_ptr1, input_ptr2, output_ptr0, output_ptr1, kernel_size, num_elements_x, num_elements_r, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements_x
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_indices_0 = x_indices
    sum_squares_0 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    sum_squares_1 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for r_offset in range(0, num_elements_r, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < num_elements_r
        r_indices_1 = r_indices
        data_0 = tl.load(input_ptr0 + (r_indices_1 + kernel_size * x_indices_0), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        data_1 = tl.load(input_ptr1 + (r_indices_1 + kernel_size * x_indices_0), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        data_2 = tl.load(input_ptr2 + (r_indices_1 + kernel_size * x_indices_0), r_mask & x_mask, eviction_policy='evict_first', other=0.0)

        diff_0 = data_0 - data_1
        epsilon = 1e-06
        diff_0_eps = diff_0 + epsilon
        squared_diff_0 = diff_0_eps * diff_0_eps
        broadcasted_squares_0 = tl.broadcast_to(squared_diff_0, [XBLOCK, RBLOCK])
        accumulated_squares_0 = sum_squares_0 + broadcasted_squares_0
        sum_squares_0 = tl.where(r_mask & x_mask, accumulated_squares_0, sum_squares_0)

        diff_1 = data_0 - data_2
        diff_1_eps = diff_1 + epsilon
        squared_diff_1 = diff_1_eps * diff_1_eps
        broadcasted_squares_1 = tl.broadcast_to(squared_diff_1, [XBLOCK, RBLOCK])
        accumulated_squares_1 = sum_squares_1 + broadcasted_squares_1
        sum_squares_1 = tl.where(r_mask & x_mask, accumulated_squares_1, sum_squares_1)

    sum_result_0 = tl.sum(sum_squares_0, 1)[:, None]
    sum_result_1 = tl.sum(sum_squares_1, 1)[:, None]
    tl.store(output_ptr0 + (x_indices_0), sum_result_0, x_mask)
    tl.store(output_ptr1 + (x_indices_0), sum_result_1, x_mask)