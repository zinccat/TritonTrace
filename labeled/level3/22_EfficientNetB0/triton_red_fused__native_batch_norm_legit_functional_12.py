# From: 22_EfficientNetB0

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_12red_fused__native_batch_norm_legit_functional_12(
    input_ptr, output_ptr_mean, output_ptr_var, output_ptr_count, 
    total_elements, reduction_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    reduction_elements = 196
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_col = (x_indices % 96)
    x_row = x_indices // 96
    mean_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    m2_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    weight_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x_flat_index = x_indices

    for reduction_offset in range(0, reduction_elements, RBLOCK):
        reduction_indices = reduction_offset + r_base
        reduction_mask = reduction_indices < reduction_elements
        reduction_index = reduction_indices
        input_data = tl.load(
            input_ptr + (x_col + 96 * reduction_index + 18816 * x_row), 
            reduction_mask, 
            eviction_policy='evict_first', 
            other=0.0
        )
        broadcast_input = tl.broadcast_to(input_data, [XBLOCK, RBLOCK])
        mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
            broadcast_input, mean_accumulator, m2_accumulator, weight_accumulator, reduction_offset == 0
        )
        mean_accumulator = tl.where(reduction_mask, mean_next, mean_accumulator)
        m2_accumulator = tl.where(reduction_mask, m2_next, m2_accumulator)
        weight_accumulator = tl.where(reduction_mask, weight_next, weight_accumulator)

    mean_result, variance_result, count_result = triton_helpers.welford(
        mean_accumulator, m2_accumulator, weight_accumulator, 1
    )
    mean_result = mean_result[:, None]
    variance_result = variance_result[:, None]
    count_result = count_result[:, None]

    tl.store(output_ptr_mean + (x_flat_index), mean_result, None)
    tl.store(output_ptr_var + (x_flat_index), variance_result, None)
    tl.store(output_ptr_count + (x_flat_index), count_result, None)