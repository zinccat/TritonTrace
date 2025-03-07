# From: 73_Conv2d_BatchNorm_Scaling

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_1(
    input_ptr, output_mean_ptr, output_var_ptr, output_weight_ptr, 
    num_elements, num_reduction_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    num_elements = 240
    num_reduction_elements = 7680
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_col = x_indices % 16
    x_row = (x_indices // 16)
    temp_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x_flat_index = x_indices

    for r_offset in range(0, num_reduction_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < num_reduction_elements
        r_flat_index = r_indices
        input_data = tl.load(
            input_ptr + ((30 * (((r_flat_index + (7680 * x_row)) // 30) % 30)) + 
                         (900 * x_col) + 
                         (14400 * ((r_flat_index + (7680 * x_row)) // 900)) + 
                         (r_flat_index % 30)), 
            r_mask & x_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )
        broadcast_input = tl.broadcast_to(input_data, [XBLOCK, RBLOCK])
        temp_mean_next, temp_m2_next, temp_weight_next = triton_helpers.welford_reduce(
            broadcast_input, temp_mean, temp_m2, temp_weight, r_offset == 0
        )
        temp_mean = tl.where(r_mask & x_mask, temp_mean_next, temp_mean)
        temp_m2 = tl.where(r_mask & x_mask, temp_m2_next, temp_m2)
        temp_weight = tl.where(r_mask & x_mask, temp_weight_next, temp_weight)

    final_mean, final_var, final_weight = triton_helpers.welford(
        temp_mean, temp_m2, temp_weight, 1
    )
    final_mean = final_mean[:, None]
    final_var = final_var[:, None]
    final_weight = final_weight[:, None]

    tl.store(output_mean_ptr + (x_flat_index), final_mean, x_mask)
    tl.store(output_var_ptr + (x_flat_index), final_var, x_mask)
    tl.store(output_weight_ptr + (x_flat_index), final_weight, x_mask)