# From: 23_EfficientNetB1

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_28(
    input_ptr, output_mean_ptr, output_var_ptr, output_count_ptr, 
    total_elements, reduction_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    total_elements = 10224
    reduction_elements = 127
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < total_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_channel = x_indices // 144
    x_within_channel = x_indices % 144
    temp_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x_flat_indices = x_indices

    for r_offset in range(0, reduction_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < reduction_elements
        r_flat_indices = r_indices
        temp_index = r_flat_indices + 127 * x_channel
        max_index = tl.full([1, 1], 9000, tl.int32)
        valid_indices = temp_index < max_index
        loaded_values = tl.load(
            input_ptr + (x_within_channel + 144 * ((temp_index % 9000))),
            r_mask & valid_indices & x_mask,
            eviction_policy='evict_first',
            other=0.0
        )
        zero_tensor = tl.full(loaded_values.shape, 0, loaded_values.dtype)
        valid_mask = tl.where(valid_indices, 0.0, zero_tensor)
        valid_count = tl.where(valid_indices, 1.0, zero_tensor)
        broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
        broadcasted_mask = tl.broadcast_to(valid_mask, [XBLOCK, RBLOCK])
        broadcasted_count = tl.broadcast_to(valid_count, [XBLOCK, RBLOCK])
        
        temp_mean_next, temp_m2_next, temp_weight_next = triton_helpers.welford_combine(
            temp_mean, temp_m2, temp_weight,
            broadcasted_values, broadcasted_mask, broadcasted_count
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
    
    tl.store(output_mean_ptr + (x_flat_indices), final_mean, x_mask)
    tl.store(output_var_ptr + (x_flat_indices), final_m2, x_mask)
    tl.store(output_count_ptr + (x_flat_indices), final_weight, x_mask)