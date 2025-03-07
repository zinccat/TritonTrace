# From: 10_ResNet101

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_12(
    input_ptr_mean, input_ptr_var, input_ptr_count, 
    output_ptr_mean, output_ptr_var, output_ptr_count, 
    total_elements, reduction_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    total_elements = 128
    reduction_elements = 123
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < total_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_channel = x_indices // 64
    x_within_channel = x_indices % 64
    temp_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x_full_indices = x_indices

    for r_offset in range(0, reduction_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < reduction_elements
        r_full_indices = r_indices
        temp_index = r_full_indices + 123 * x_channel
        max_index = tl.full([1, 1], 245, tl.int32)
        index_mask = temp_index < max_index
        temp_data_mean = tl.load(
            input_ptr_mean + (x_within_channel + 64 * r_full_indices + 7872 * x_channel), 
            r_mask & index_mask & x_mask, 
            eviction_policy='evict_first', 
            other=0.0
        )
        temp_data_var = tl.load(
            input_ptr_var + (x_within_channel + 64 * r_full_indices + 7872 * x_channel), 
            r_mask & index_mask & x_mask, 
            eviction_policy='evict_first', 
            other=0.0
        )
        temp_data_count = tl.load(
            input_ptr_count + (x_within_channel + 64 * r_full_indices + 7872 * x_channel), 
            r_mask & index_mask & x_mask, 
            eviction_policy='evict_first', 
            other=0.0
        )
        temp_broadcast_mean = tl.broadcast_to(temp_data_mean, [XBLOCK, RBLOCK])
        temp_broadcast_var = tl.broadcast_to(temp_data_var, [XBLOCK, RBLOCK])
        temp_broadcast_count = tl.broadcast_to(temp_data_count, [XBLOCK, RBLOCK])
        
        temp_mean_next, temp_m2_next, temp_weight_next = triton_helpers.welford_combine(
            temp_mean, temp_m2, temp_weight,
            temp_broadcast_mean, temp_broadcast_var, temp_broadcast_count
        )
        
        temp_mean = tl.where(r_mask & x_mask, temp_mean_next, temp_mean)
        temp_m2 = tl.where(r_mask & x_mask, temp_m2_next, temp_m2)
        temp_weight = tl.where(r_mask & x_mask, temp_weight_next, temp_weight)

    final_mean, final_m2, final_weight = triton_helpers.welford(
        temp_mean, temp_m2, temp_weight, 1
    )
    
    final_mean = final_mean[:, None]
    final_m2 = final_m2[:, None]
    final_weight = final_weight[:, None]
    
    tl.store(output_ptr_mean + (x_full_indices), final_mean, x_mask)
    tl.store(output_ptr_var + (x_full_indices), final_m2, x_mask)
    tl.store(output_ptr_count + (x_full_indices), final_weight, x_mask)