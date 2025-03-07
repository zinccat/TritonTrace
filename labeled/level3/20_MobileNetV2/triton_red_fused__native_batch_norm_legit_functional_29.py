# From: 20_MobileNetV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_29red_fused__native_batch_norm_legit_functional_29(
    input_ptr, output_mean_ptr, output_var_ptr, output_weight_ptr, 
    total_elements, reduction_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    total_elements = 1984
    reduction_elements = 127
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < total_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_channel = x_indices // 32
    x_within_channel = (x_indices % 32)
    tmp_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x_full_indices = x_indices

    for r_offset in range(0, reduction_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < reduction_elements
        r_full_indices = r_indices
        tmp_combined_index = r_full_indices + 127 * x_channel
        max_index = tl.full([1, 1], 7840, tl.int32)
        index_within_limit = tmp_combined_index < max_index
        loaded_values = tl.load(
            input_ptr + (x_within_channel + 32 * ((tmp_combined_index % 7840))),
            r_mask & index_within_limit & x_mask,
            eviction_policy='evict_first',
            other=0.0
        )
        zero_tensor = tl.full(loaded_values.shape, 0, loaded_values.dtype)
        mask_tensor = tl.where(index_within_limit, 0.0, zero_tensor)
        one_tensor = tl.full(loaded_values.shape, 0, 1.0)
        broadcast_one = tl.where(index_within_limit, 1.0, one_tensor)
        
        broadcast_loaded_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
        broadcast_mask_tensor = tl.broadcast_to(mask_tensor, [XBLOCK, RBLOCK])
        broadcast_broadcast_one = tl.broadcast_to(broadcast_one, [XBLOCK, RBLOCK])
        
        tmp_mean_next, tmp_m2_next, tmp_weight_next = triton_helpers.welford_combine(
            tmp_mean, tmp_m2, tmp_weight,
            broadcast_loaded_values, broadcast_mask_tensor, broadcast_broadcast_one
        )
        
        tmp_mean = tl.where(r_mask & x_mask, tmp_mean_next, tmp_mean)
        tmp_m2 = tl.where(r_mask & x_mask, tmp_m2_next, tmp_m2)
        tmp_weight = tl.where(r_mask & x_mask, tmp_weight_next, tmp_weight)

    tmp_mean_final, tmp_m2_final, tmp_weight_final = triton_helpers.welford(
        tmp_mean, tmp_m2, tmp_weight, 1
    )
    
    tmp_mean_final = tmp_mean_final[:, None]
    tmp_m2_final = tmp_m2_final[:, None]
    tmp_weight_final = tmp_weight_final[:, None]
    
    tl.store(output_mean_ptr + (x_full_indices), tmp_mean_final, x_mask)
    tl.store(output_var_ptr + (x_full_indices), tmp_m2_final, x_mask)
    tl.store(output_weight_ptr + (x_full_indices), tmp_weight_final, x_mask)