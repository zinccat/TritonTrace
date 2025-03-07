# From: 22_EfficientNetB0

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_62red_fused__native_batch_norm_legit_functional_62(
    input_ptr, output_mean_ptr, output_var_ptr, output_weight_ptr, 
    total_elements, reduction_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    total_elements = 4608
    reduction_elements = 123
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < total_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_channel = x_indices // 1152
    x_within_channel = x_indices % 1152
    temp_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x_full_index = x_indices

    for r_offset in range(0, reduction_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < reduction_elements
        r_index = r_indices
        temp_index = r_index + 123 * x_channel
        max_index = tl.full([1, 1], 490, tl.int32)
        valid_mask = temp_index < max_index
        temp_data = tl.load(
            input_ptr + (x_within_channel + 1152 * ((temp_index % 490))),
            valid_mask & x_mask,
            eviction_policy='evict_first',
            other=0.0
        )
        zero_tensor = tl.full(temp_data.shape, 0, temp_data.dtype)
        valid_data = tl.where(valid_mask, temp_data, zero_tensor)
        valid_weight = tl.where(valid_mask, 1.0, 0.0)
        broadcast_data = tl.broadcast_to(valid_data, [XBLOCK, RBLOCK])
        broadcast_zero = tl.broadcast_to(zero_tensor, [XBLOCK, RBLOCK])
        broadcast_weight = tl.broadcast_to(valid_weight, [XBLOCK, RBLOCK])

        temp_mean_next, temp_m2_next, temp_weight_next = triton_helpers.welford_combine(
            temp_mean, temp_m2, temp_weight,
            broadcast_data, broadcast_zero, broadcast_weight
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

    tl.store(output_mean_ptr + (x_full_index), final_mean, x_mask)
    tl.store(output_var_ptr + (x_full_index), final_m2, x_mask)
    tl.store(output_weight_ptr + (x_full_index), final_weight, x_mask)