# From: 33_BatchNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_0(
    input_ptr, output_mean_ptr, output_var_ptr, output_weight_ptr, 
    num_elements, num_reduction_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    num_elements = 384
    num_reduction_elements = 174763
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_div_64 = x_indices // 64
    x_mod_64 = x_indices % 64
    temp_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x_indices_flat = x_indices

    for r_offset in range(0, num_reduction_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < num_reduction_elements
        r_indices_flat = r_indices
        temp_index = r_indices_flat + (num_reduction_elements * x_div_64)
        max_index = tl.full([1, 1], 1048576, tl.int32)
        index_mask = temp_index < max_index
        temp_data = tl.load(
            input_ptr + ((65536 * x_mod_64) + (4194304 * (((r_indices_flat + (num_reduction_elements * x_div_64)) // 65536) % 16)) + ((r_indices_flat + (num_reduction_elements * x_div_64)) % 65536)),
            r_mask & index_mask & x_mask,
            eviction_policy='evict_last',
            other=0.0
        )
        zero_value = 0.0
        zero_broadcast = tl.full(zero_value.shape, 0, zero_value.dtype)
        temp_zero = tl.where(index_mask, zero_value, zero_broadcast)
        one_value = 1.0
        one_broadcast = tl.full(one_value.shape, 0, one_value.dtype)
        temp_one = tl.where(index_mask, one_value, one_broadcast)
        temp_data_broadcast = tl.broadcast_to(temp_data, [XBLOCK, RBLOCK])
        temp_zero_broadcast = tl.broadcast_to(temp_zero, [XBLOCK, RBLOCK])
        temp_one_broadcast = tl.broadcast_to(temp_one, [XBLOCK, RBLOCK])
        temp_mean_next, temp_m2_next, temp_weight_next = triton_helpers.welford_combine(
            temp_mean, temp_m2, temp_weight,
            temp_data_broadcast, temp_zero_broadcast, temp_one_broadcast
        )
        temp_mean = tl.where(r_mask & x_mask, temp_mean_next, temp_mean)
        temp_m2 = tl.where(r_mask & x_mask, temp_m2_next, temp_m2)
        temp_weight = tl.where(r_mask & x_mask, temp_weight_next, temp_weight)

    final_mean, final_m2, final_weight = triton_helpers.welford(
        temp_mean, temp_m2, temp_weight, 1
    )
    final_mean_broadcast = final_mean[:, None]
    final_m2_broadcast = final_m2[:, None]
    final_weight_broadcast = final_weight[:, None]
    tl.store(output_mean_ptr + (x_indices_flat), final_mean_broadcast, x_mask)
    tl.store(output_var_ptr + (x_indices_flat), final_m2_broadcast, x_mask)
    tl.store(output_weight_ptr + (x_indices_flat), final_weight_broadcast, x_mask)