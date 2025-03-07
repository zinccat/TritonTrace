# From: 85_Conv2d_GroupNorm_Scale_MaxPool_Clamp

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_group_norm_1(input_ptr, output_mean_ptr, output_var_ptr, output_scale_ptr, kernel_size, input_num_elements, reduction_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    input_offset = tl.program_id(0) * XBLOCK
    input_indices = input_offset + tl.arange(0, XBLOCK)[:, None]
    input_mask = input_indices < input_num_elements
    reduction_base = tl.arange(0, RBLOCK)[None, :]
    input_indices_flat = input_indices
    running_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)

    for reduction_offset in range(0, reduction_num_elements, RBLOCK):
        reduction_indices = reduction_offset + reduction_base
        reduction_mask = reduction_indices < reduction_num_elements
        reduction_indices_flat = reduction_indices
        input_data = tl.load(input_ptr + (reduction_indices_flat + 8 * input_indices_flat + ((-8) * kernel_size * input_indices_flat) + 2 * input_indices_flat * kernel_size * kernel_size), reduction_mask & input_mask, eviction_policy='evict_first', other=0.0)
        broadcasted_input = tl.broadcast_to(input_data, [XBLOCK, RBLOCK])
        running_mean_next, running_m2_next, running_weight_next = triton_helpers.welford_reduce(
            broadcasted_input, running_mean, running_m2, running_weight, reduction_offset == 0
        )
        running_mean = tl.where(reduction_mask & input_mask, running_mean_next, running_mean)
        running_m2 = tl.where(reduction_mask & input_mask, running_m2_next, running_m2)
        running_weight = tl.where(reduction_mask & input_mask, running_weight_next, running_weight)

    mean, variance, weight = triton_helpers.welford(running_mean, running_m2, running_weight, 1)
    mean_broadcasted = mean[:, None]
    variance_broadcasted = variance[:, None]
    weight_broadcasted = weight[:, None]

    tl.store(output_mean_ptr + (input_indices_flat), mean_broadcasted, input_mask)
    tl.store(output_var_ptr + (input_indices_flat), variance_broadcasted, input_mask)

    epsilon_offset = (tl.full([], 0.0, tl.float64) >= (8 + ((-8) * kernel_size) + 2 * kernel_size * kernel_size)) * 0.0 + (8 + ((-8) * kernel_size) + 2 * kernel_size * kernel_size) * ((8 + ((-8) * kernel_size) + 2 * kernel_size * kernel_size) > (tl.full([], 0.0, tl.float64)))
    epsilon_offset = epsilon_offset.to(tl.float32)
    variance_adjusted = variance_broadcasted / epsilon_offset
    epsilon = 1e-05
    variance_stabilized = variance_adjusted + epsilon
    inverse_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_stabilized)

    tl.store(output_scale_ptr + (input_indices_flat), inverse_sqrt_variance, input_mask)