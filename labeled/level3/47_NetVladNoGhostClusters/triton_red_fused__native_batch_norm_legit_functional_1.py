# From: 47_NetVladNoGhostClusters

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_1red_fused__native_batch_norm_legit_functional_1(
    input_ptr, output_ptr_mean, output_ptr_var, output_ptr_weight, 
    num_elements_x, num_elements_r, BLOCK_SIZE_X : tl.constexpr, BLOCK_SIZE_R : tl.constexpr
):
    num_elements_x = 800
    num_elements_r = 128
    x_offset = tl.program_id(0) * BLOCK_SIZE_X
    x_indices = x_offset + tl.arange(0, BLOCK_SIZE_X)[:, None]
    x_mask = x_indices < num_elements_x
    r_base_indices = tl.arange(0, BLOCK_SIZE_R)[None, :]
    x_index_mod_32 = (x_indices % 32)
    x_index_div_32 = x_indices // 32
    running_mean = tl.zeros([BLOCK_SIZE_X, BLOCK_SIZE_R], tl.float32)
    running_m2 = tl.zeros([BLOCK_SIZE_X, BLOCK_SIZE_R], tl.float32)
    running_weight = tl.zeros([BLOCK_SIZE_X, BLOCK_SIZE_R], tl.float32)
    x_indices_flat = x_indices

    for r_offset in range(0, num_elements_r, BLOCK_SIZE_R):
        r_indices = r_offset + r_base_indices
        r_mask = r_indices < num_elements_r
        r_indices_flat = r_indices
        input_data = tl.load(
            input_ptr + (x_index_mod_32 + 32 * r_indices_flat + 4096 * x_index_div_32), 
            r_mask & x_mask, 
            eviction_policy='evict_first', 
            other=0.0
        )
        broadcast_input = tl.broadcast_to(input_data, [BLOCK_SIZE_X, BLOCK_SIZE_R])
        running_mean_next, running_m2_next, running_weight_next = triton_helpers.welford_reduce(
            broadcast_input, running_mean, running_m2, running_weight, r_offset == 0
        )
        running_mean = tl.where(r_mask & x_mask, running_mean_next, running_mean)
        running_m2 = tl.where(r_mask & x_mask, running_m2_next, running_m2)
        running_weight = tl.where(r_mask & x_mask, running_weight_next, running_weight)

    final_mean, final_var, final_weight = triton_helpers.welford(
        running_mean, running_m2, running_weight, 1
    )
    final_mean = final_mean[:, None]
    final_var = final_var[:, None]
    final_weight = final_weight[:, None]

    tl.store(output_ptr_mean + (x_indices_flat), final_mean, x_mask)
    tl.store(output_ptr_var + (x_indices_flat), final_var, x_mask)
    tl.store(output_ptr_weight + (x_indices_flat), final_weight, x_mask)