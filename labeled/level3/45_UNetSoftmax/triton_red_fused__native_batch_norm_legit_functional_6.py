# From: 45_UNetSoftmax

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_6red_fused__native_batch_norm_legit_functional_6(
    input_ptr, output_mean_ptr, output_var_ptr, output_weight_ptr, 
    num_elements_x, num_elements_r, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    num_elements_x = 512
    num_elements_r = 16384
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements_x
    r_base_indices = tl.arange(0, RBLOCK)[None, :]
    x_index_mod_128 = x_indices % 128
    x_index_div_128 = x_indices // 128
    temp_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x_indices_flat = x_indices

    for r_offset in range(0, num_elements_r, RBLOCK):
        r_indices = r_offset + r_base_indices
        r_mask = r_indices < num_elements_r
        r_indices_flat = r_indices
        temp_input = tl.load(
            input_ptr + (8192 * x_index_mod_128 + 1048576 * (r_indices_flat // 8192) + 2097152 * x_index_div_128 + ((r_indices_flat % 8192))),
            r_mask & x_mask, eviction_policy='evict_first', other=0.0
        )
        temp_broadcast = tl.broadcast_to(temp_input, [XBLOCK, RBLOCK])
        temp_mean_next, temp_m2_next, temp_weight_next = triton_helpers.welford_reduce(
            temp_broadcast, temp_mean, temp_m2, temp_weight, r_offset == 0
        )
        temp_mean = tl.where(r_mask & x_mask, temp_mean_next, temp_mean)
        temp_m2 = tl.where(r_mask & x_mask, temp_m2_next, temp_m2)
        temp_weight = tl.where(r_mask & x_mask, temp_weight_next, temp_weight)

    temp_mean_final, temp_var_final, temp_weight_final = triton_helpers.welford(
        temp_mean, temp_m2, temp_weight, 1
    )
    temp_mean_final = temp_mean_final[:, None]
    temp_var_final = temp_var_final[:, None]
    temp_weight_final = temp_weight_final[:, None]

    tl.store(output_mean_ptr + (x_indices_flat), temp_mean_final, x_mask)
    tl.store(output_var_ptr + (x_indices_flat), temp_var_final, x_mask)
    tl.store(output_weight_ptr + (x_indices_flat), temp_weight_final, x_mask)