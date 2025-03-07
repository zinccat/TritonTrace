# From: 23_Conv3d_GroupNorm_Mean

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_group_norm_1red_fused_native_group_norm_1(
    input_ptr, output_mean_ptr, output_var_ptr, output_scale_ptr, kernel_size_0, kernel_size_1, 
    num_elements_x, num_elements_r, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements_x
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_indices_flat = x_indices
    running_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)

    for r_offset in range(0, num_elements_r, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < num_elements_r
        r_indices_flat = r_indices
        input_data = tl.load(
            input_ptr + (r_indices_flat + ((-16) * x_indices_flat) + ((-4) * x_indices_flat * kernel_size_1 * kernel_size_1) + 8 * kernel_size_0 * x_indices_flat + 16 * kernel_size_1 * x_indices_flat + ((-8) * kernel_size_0 * kernel_size_1 * x_indices_flat) + 2 * kernel_size_0 * x_indices_flat * kernel_size_1 * kernel_size_1),
            r_mask & x_mask,
            eviction_policy='evict_first',
            other=0.0
        )
        broadcast_input = tl.broadcast_to(input_data, [XBLOCK, RBLOCK])
        running_mean_next, running_m2_next, running_weight_next = triton_helpers.welford_reduce(
            broadcast_input, running_mean, running_m2, running_weight, r_offset == 0
        )
        running_mean = tl.where(r_mask & x_mask, running_mean_next, running_mean)
        running_m2 = tl.where(r_mask & x_mask, running_m2_next, running_m2)
        running_weight = tl.where(r_mask & x_mask, running_weight_next, running_weight)

    mean, variance, weight = triton_helpers.welford(running_mean, running_m2, running_weight, 1)
    mean_broadcast = mean[:, None]
    variance_broadcast = variance[:, None]
    weight_broadcast = weight[:, None]

    tl.store(output_mean_ptr + (x_indices_flat), mean_broadcast, x_mask)
    tl.store(output_var_ptr + (x_indices_flat), variance_broadcast, x_mask)

    offset_calculation = (
        (tl.full([], 0.0, tl.float64) * (tl.full([], 0.0, tl.float64) >= ((-16) + ((-4) * kernel_size_1 * kernel_size_1) + 8 * kernel_size_0 + 16 * kernel_size_1 + ((-8) * kernel_size_0 * kernel_size_1) + 2 * kernel_size_0 * kernel_size_1 * kernel_size_1))) +
        ((-16) + ((-4) * kernel_size_1 * kernel_size_1) + 8 * kernel_size_0 + 16 * kernel_size_1 + ((-8) * kernel_size_0 * kernel_size_1) + 2 * kernel_size_0 * kernel_size_1 * kernel_size_1) * 
        (((-16) + ((-4) * kernel_size_1 * kernel_size_1) + 8 * kernel_size_0 + 16 * kernel_size_1 + ((-8) * kernel_size_0 * kernel_size_1) + 2 * kernel_size_0 * kernel_size_1 * kernel_size_1) > (tl.full([], 0.0, tl.float64)))
    )
    offset_float32 = offset_calculation.to(tl.float32)
    variance_adjusted = variance_broadcast / offset_float32
    epsilon = 1e-05
    variance_adjusted_epsilon = variance_adjusted + epsilon
    scale_factor = tl.extra.cuda.libdevice.rsqrt(variance_adjusted_epsilon)

    tl.store(output_scale_ptr + (x_indices_flat), scale_factor, x_mask)