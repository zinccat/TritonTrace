# From: 10_ResNet101

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_37(
    input_ptr, output_mean_ptr, output_var_ptr, output_count_ptr, 
    xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    rnumel = 123
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_base = tl.arange(0, RBLOCK)[None, :]
    x1 = x_index // 1024
    x0 = (x_index % 1024)
    running_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = x_index

    for r_offset in range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel
        r2 = r_index
        tmp_index = r2 + 123 * x1
        max_index = tl.full([1, 1], 1960, tl.int32)
        index_mask = tmp_index < max_index
        loaded_values = tl.load(
            input_ptr + (x0 + 1024 * ((r2 + 123 * x1) % 1960)), 
            r_mask & index_mask, 
            eviction_policy='evict_first', 
            other=0.0
        )
        zero_values = tl.full(loaded_values.shape, 0, loaded_values.dtype)
        mask_values = tl.where(index_mask, 0.0, zero_values)
        one_values = tl.full(loaded_values.shape, 0, 1.0)
        one_mask = tl.where(index_mask, 1.0, one_values)

        broadcasted_values = tl.broadcast_to(loaded_values, [XBLOCK, RBLOCK])
        broadcasted_mask_values = tl.broadcast_to(mask_values, [XBLOCK, RBLOCK])
        broadcasted_one_mask = tl.broadcast_to(one_mask, [XBLOCK, RBLOCK])

        running_mean_next, running_m2_next, running_weight_next = triton_helpers.welford_combine(
            running_mean, running_m2, running_weight,
            broadcasted_values, broadcasted_mask_values, broadcasted_one_mask
        )

        running_mean = tl.where(r_mask, running_mean_next, running_mean)
        running_m2 = tl.where(r_mask, running_m2_next, running_m2)
        running_weight = tl.where(r_mask, running_weight_next, running_weight)

    final_mean, final_m2, final_weight = triton_helpers.welford(
        running_mean, running_m2, running_weight, 1
    )

    final_mean = final_mean[:, None]
    final_m2 = final_m2[:, None]
    final_weight = final_weight[:, None]

    tl.store(output_mean_ptr + (x3), final_mean, None)
    tl.store(output_var_ptr + (x3), final_m2, None)
    tl.store(output_count_ptr + (x3), final_weight, None)