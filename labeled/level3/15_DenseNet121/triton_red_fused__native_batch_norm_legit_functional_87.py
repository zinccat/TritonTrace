# From: 15_DenseNet121

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_87(
    input_ptr_mean, input_ptr_var, input_ptr_input, 
    output_ptr_mean, output_ptr_var, output_ptr_input, 
    output_ptr_normalized, output_ptr_scale, 
    num_elements_input, num_elements_reduction, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    num_elements_input = 384
    num_elements_reduction = 1960
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < num_elements_input
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_indices = input_index

    running_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)

    for reduction_offset in range(0, num_elements_reduction, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < num_elements_reduction
        reduction_index_1 = (reduction_index % 196)
        reduction_index_2 = reduction_index // 196

        input_data = tl.load(
            input_ptr_input + (reduction_index_1 + 196 * input_indices + 75264 * reduction_index_2), 
            reduction_mask & input_mask, 
            eviction_policy='evict_first', 
            other=0.0
        )
        broadcast_input = tl.broadcast_to(input_data, [XBLOCK, RBLOCK])

        running_mean_next, running_m2_next, running_weight_next = triton_helpers.welford_reduce(
            broadcast_input, running_mean, running_m2, running_weight, reduction_offset == 0
        )

        running_mean = tl.where(reduction_mask & input_mask, running_mean_next, running_mean)
        running_m2 = tl.where(reduction_mask & input_mask, running_m2_next, running_m2)
        running_weight = tl.where(reduction_mask & input_mask, running_weight_next, running_weight)

    final_mean, final_var, _ = triton_helpers.welford(
        running_mean, running_m2, running_weight, 1
    )

    final_mean = final_mean[:, None]
    final_var = final_var[:, None]

    tl.store(output_ptr_mean + (input_indices), final_mean, input_mask)
    tl.store(output_ptr_var + (input_indices), final_var, input_mask)

    input_mean = tl.load(input_ptr_mean + (input_indices), input_mask, eviction_policy='evict_last')
    input_var = tl.load(input_ptr_var + (input_indices), input_mask, eviction_policy='evict_last')

    num_reduction_elements = 1960.0
    mean_div_num_elements = final_var / num_reduction_elements
    epsilon = 1e-05
    variance_plus_epsilon = mean_div_num_elements + epsilon
    reciprocal_sqrt = tl.extra.cuda.libdevice.rsqrt(variance_plus_epsilon)

    momentum = 0.1
    running_mean_scaled = final_mean * momentum
    input_mean_scaled = input_mean * 0.9
    updated_mean = running_mean_scaled + input_mean_scaled

    variance_scale = 1.0005104645227156
    variance_scaled = mean_div_num_elements * variance_scale
    variance_scaled_with_momentum = variance_scaled * momentum

    updated_var = variance_scaled_with_momentum + input_var * 0.9

    tl.store(output_ptr_normalized + (input_indices), reciprocal_sqrt, input_mask)
    tl.store(output_ptr_scale + (input_indices), updated_mean, input_mask)
    tl.store(output_ptr_input + (input_indices), updated_var, input_mask)