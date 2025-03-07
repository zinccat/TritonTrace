# From: 20_MobileNetV2

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_10red_fused__native_batch_norm_legit_functional_10(
    input_ptr, output_ptr_mean, output_ptr_var, output_ptr_count, 
    total_elements, reduction_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    reduction_elements = 196
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_channel = (x_indices % 96)
    x_batch = x_indices // 96
    running_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x_flat_index = x_indices

    for reduction_offset in range(0, reduction_elements, RBLOCK):
        reduction_indices = reduction_offset + r_base
        reduction_mask = reduction_indices < reduction_elements
        reduction_index = reduction_indices
        input_data = tl.load(
            input_ptr + (x_channel + 96 * reduction_index + 18816 * x_batch), 
            reduction_mask, 
            eviction_policy='evict_first', 
            other=0.0
        )
        broadcast_input = tl.broadcast_to(input_data, [XBLOCK, RBLOCK])
        running_mean_next, running_m2_next, running_weight_next = triton_helpers.welford_reduce(
            broadcast_input, running_mean, running_m2, running_weight, reduction_offset == 0
        )
        running_mean = tl.where(reduction_mask, running_mean_next, running_mean)
        running_m2 = tl.where(reduction_mask, running_m2_next, running_m2)
        running_weight = tl.where(reduction_mask, running_weight_next, running_weight)

    final_mean, final_var, final_count = triton_helpers.welford(
        running_mean, running_m2, running_weight, 1
    )
    final_mean = final_mean[:, None]
    final_var = final_var[:, None]
    final_count = final_count[:, None]

    tl.store(output_ptr_mean + (x_flat_index), final_mean, None)
    tl.store(output_ptr_var + (x_flat_index), final_var, None)
    tl.store(output_ptr_count + (x_flat_index), final_count, None)