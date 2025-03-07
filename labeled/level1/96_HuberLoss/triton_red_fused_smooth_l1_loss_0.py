# From: 96_HuberLoss

import triton
import triton.language as tl


@triton.jit
def triton_red_fused_smooth_l1_loss_0(input_ptr0, input_ptr1, output_ptr0, num_elements_x, num_elements_r, BLOCK_SIZE_X : tl.constexpr, BLOCK_SIZE_R : tl.constexpr):
    num_elements_x = 64
    num_elements_r = 8192
    x_offset = tl.program_id(0) * BLOCK_SIZE_X
    x_indices = x_offset + tl.arange(0, BLOCK_SIZE_X)[:, None]
    x_mask = x_indices < num_elements_x
    r_base = tl.arange(0, BLOCK_SIZE_R)[None, :]
    x_indices_0 = x_indices
    temp_sum = tl.full([BLOCK_SIZE_X, BLOCK_SIZE_R], 0, tl.float32)
    
    for r_offset in range(0, num_elements_r, BLOCK_SIZE_R):
        r_indices = r_offset + r_base
        r_mask = r_indices < num_elements_r
        r_indices_1 = r_indices
        temp_diff_0 = tl.load(input_ptr0 + (r_indices_1 + (8192 * x_indices_0)), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        temp_diff_1 = tl.load(input_ptr1 + (r_indices_1 + (8192 * x_indices_0)), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        temp_subtraction = temp_diff_0 - temp_diff_1
        temp_abs = tl.math.abs(temp_subtraction)
        threshold = 1.0
        is_below_threshold = temp_abs < threshold
        temp_squared = temp_abs * temp_abs
        half = 0.5
        temp_scaled_squared = temp_squared * half * threshold
        temp_adjusted = temp_abs - half
        temp_result = tl.where(is_below_threshold, temp_scaled_squared, temp_adjusted)
        temp_broadcasted = tl.broadcast_to(temp_result, [BLOCK_SIZE_X, BLOCK_SIZE_R])
        temp_accumulated = temp_sum + temp_broadcasted
        temp_sum = tl.where(r_mask & x_mask, temp_accumulated, temp_sum)
    
    temp_final_sum = tl.sum(temp_sum, 1)[:, None]
    tl.store(output_ptr0 + (x_indices_0), temp_final_sum, x_mask)