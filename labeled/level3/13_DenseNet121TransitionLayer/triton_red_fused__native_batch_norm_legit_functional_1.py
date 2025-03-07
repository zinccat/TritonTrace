# From: 13_DenseNet121TransitionLayer

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_1red_fused__native_batch_norm_legit_functional_1(
    input_ptr, output_mean_ptr, output_var_ptr, output_weight_ptr, 
    total_elements, reduction_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    total_elements = 320
    reduction_elements = 50176
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < total_elements
    r_base_indices = tl.arange(0, RBLOCK)[None, :]
    x_indices_flat = x_indices
    mean_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    variance_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    weight_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)

    for r_offset in range(0, reduction_elements, RBLOCK):
        r_indices = r_offset + r_base_indices
        r_mask = r_indices < reduction_elements
        r_indices_flat = r_indices
        input_values = tl.load(input_ptr + (r_indices_flat + 50176 * x_indices_flat), r_mask & x_mask, eviction_policy='evict_first', other=0.0)
        broadcasted_values = tl.broadcast_to(input_values, [XBLOCK, RBLOCK])
        mean_next, variance_next, weight_next = triton_helpers.welford_reduce(
            broadcasted_values, mean_accumulator, variance_accumulator, weight_accumulator, r_offset == 0
        )
        mean_accumulator = tl.where(r_mask & x_mask, mean_next, mean_accumulator)
        variance_accumulator = tl.where(r_mask & x_mask, variance_next, variance_accumulator)
        weight_accumulator = tl.where(r_mask & x_mask, weight_next, weight_accumulator)

    mean_result, variance_result, weight_result = triton_helpers.welford(
        mean_accumulator, variance_accumulator, weight_accumulator, 1
    )
    mean_result_broadcasted = mean_result[:, None]
    variance_result_broadcasted = variance_result[:, None]
    weight_result_broadcasted = weight_result[:, None]

    tl.store(output_mean_ptr + (x_indices_flat), mean_result_broadcasted, x_mask)
    tl.store(output_var_ptr + (x_indices_flat), variance_result_broadcasted, x_mask)
    tl.store(output_weight_ptr + (x_indices_flat), weight_result_broadcasted, x_mask)