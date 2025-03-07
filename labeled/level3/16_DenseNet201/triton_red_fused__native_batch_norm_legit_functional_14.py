# From: 16_DenseNet201

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_14red_fused__native_batch_norm_legit_functional_14(
    input_ptr, output_mean_ptr, output_var_ptr, output_weight_ptr, 
    num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    num_elements = 512
    reduction_num_elements = 7840
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_col = (x_indices % 128)
    x_row = x_indices // 128
    running_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x_flat_indices = x_indices

    for r_offset in range(0, reduction_num_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < reduction_num_elements
        r_flat_indices = r_indices
        input_data = tl.load(
            input_ptr + (56 * (((r_flat_indices + reduction_num_elements * x_row) // 56) % 56) + 
                         3136 * x_col + 401408 * ((r_flat_indices + reduction_num_elements * x_row) // 3136) + 
                         (r_flat_indices % 56)), 
            r_mask & x_mask, eviction_policy='evict_last', other=0.0
        )
        broadcasted_data = tl.broadcast_to(input_data, [XBLOCK, RBLOCK])
        running_mean_next, running_m2_next, running_weight_next = triton_helpers.welford_reduce(
            broadcasted_data, running_mean, running_m2, running_weight, r_offset == 0
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

    tl.store(output_mean_ptr + (x_flat_indices), final_mean, x_mask)
    tl.store(output_var_ptr + (x_flat_indices), final_var, x_mask)
    tl.store(output_weight_ptr + (x_flat_indices), final_weight, x_mask)