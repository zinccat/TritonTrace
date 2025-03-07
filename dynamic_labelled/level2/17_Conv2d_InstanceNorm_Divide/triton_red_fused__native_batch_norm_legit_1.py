# From: 17_Conv2d_InstanceNorm_Divide

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_1red_fused__native_batch_norm_legit_1(
    input_ptr, output_mean_ptr, output_var_ptr, output_scale_ptr, kernel_size, input_num_elements, reduction_num_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    input_offset = tl.program_id(0) * XBLOCK
    input_index = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_index < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_indices = input_index
    running_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_index = reduction_offset + reduction_base
        reduction_mask = reduction_index < reduction_num_elements
        reduction_indices = reduction_index
        input_data = tl.load(
            input_ptr + (reduction_indices + 4 * input_indices + input_indices * kernel_size * kernel_size + ((-4) * kernel_size * input_indices)), 
            reduction_mask & input_mask, 
            eviction_policy='evict_first', 
            other=0.0
        )
        broadcasted_data = tl.broadcast_to(input_data, [XBLOCK, RBLOCK])
        running_mean_next, running_m2_next, running_weight_next = triton_helpers.welford_reduce(
            broadcasted_data, running_mean, running_m2, running_weight, reduction_offset == 0
        )
        running_mean = tl.where(reduction_mask & input_mask, running_mean_next, running_mean)
        running_m2 = tl.where(reduction_mask & input_mask, running_m2_next, running_m2)
        running_weight = tl.where(reduction_mask & input_mask, running_weight_next, running_weight)

    mean, variance, weight = triton_helpers.welford(running_mean, running_m2, running_weight, 1)
    mean = mean[:, None]
    variance = variance[:, None]
    weight = weight[:, None]

    tl.store(output_mean_ptr + (input_indices), mean, input_mask)
    tl.store(output_var_ptr + (input_indices), variance, input_mask)

    scale_factor = 4 + kernel_size * kernel_size + ((-4) * kernel_size)
    scale_factor_float = scale_factor.to(tl.float32)
    normalized_variance = variance / scale_factor_float
    epsilon = 1e-05
    adjusted_variance = normalized_variance + epsilon
    reciprocal_sqrt = tl.extra.cuda.libdevice.rsqrt(adjusted_variance)

    tl.store(output_scale_ptr + (input_indices), reciprocal_sqrt, input_mask)