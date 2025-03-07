# From: 20_MobileNetV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_6red_fused__native_batch_norm_legit_functional_6(
    input_ptr, output_ptr_mean, output_ptr_var, output_ptr_weight, 
    total_elements, reduction_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    total_elements = 15680
    reduction_elements = 128
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < total_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_channel = (x_indices % 16)
    x_height = x_indices // 16
    temp_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    temp_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x_flat_index = x_indices

    for r_offset in range(0, reduction_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < reduction_elements
        r_flat_index = r_indices
        temp_input = tl.load(
            input_ptr + (x_channel + 16 * r_flat_index + 2048 * x_height), 
            r_mask & x_mask, 
            eviction_policy='evict_first', 
            other=0.0
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

    tl.store(output_ptr_mean + (x_flat_index), temp_mean_final, x_mask)
    tl.store(output_ptr_var + (x_flat_index), temp_var_final, x_mask)
    tl.store(output_ptr_weight + (x_flat_index), temp_weight_final, x_mask)