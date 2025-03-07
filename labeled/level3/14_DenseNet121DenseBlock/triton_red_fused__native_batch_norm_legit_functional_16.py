# From: 14_DenseNet121DenseBlock

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_16red_fused__native_batch_norm_legit_functional_16(
    input_ptr, output_mean_ptr, output_var_ptr, output_weight_ptr, 
    total_elements, reduction_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    total_elements = 640
    reduction_elements = 125440
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < total_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_col = (x_indices % 160)
    x_row = x_indices // 160
    mean_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    m2_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    weight_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x_flat_index = x_indices

    for r_offset in range(0, reduction_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < reduction_elements
        r_flat_index = r_indices
        input_data = tl.load(
            input_ptr + (224 * (((r_flat_index + reduction_elements * x_row) // 224) % 224) + 
                         50176 * x_col + 
                         8028160 * ((r_flat_index + reduction_elements * x_row) // 50176) + 
                         (r_flat_index % 224)),
            r_mask & x_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )
        broadcasted_data = tl.broadcast_to(input_data, [XBLOCK, RBLOCK])
        mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
            broadcasted_data, mean_accumulator, m2_accumulator, weight_accumulator, r_offset == 0
        )
        mean_accumulator = tl.where(r_mask & x_mask, mean_next, mean_accumulator)
        m2_accumulator = tl.where(r_mask & x_mask, m2_next, m2_accumulator)
        weight_accumulator = tl.where(r_mask & x_mask, weight_next, weight_accumulator)

    mean_result, variance_result, weight_result = triton_helpers.welford(
        mean_accumulator, m2_accumulator, weight_accumulator, 1
    )
    mean_result = mean_result[:, None]
    variance_result = variance_result[:, None]
    weight_result = weight_result[:, None]

    tl.store(output_mean_ptr + (x_flat_index), mean_result, x_mask)
    tl.store(output_var_ptr + (x_flat_index), variance_result, x_mask)
    tl.store(output_weight_ptr + (x_flat_index), weight_result, x_mask)