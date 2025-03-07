# From: 35_GroupNorm_

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_per_fused_native_group_norm_1(input_ptr_mean, input_ptr_var, input_ptr_count, output_ptr_mean, output_ptr_var, output_ptr_count, kernel_size, num_elements, num_reduction_elements, XBLOCK: tl.constexpr):
    num_reduction_elements = 3
    RBLOCK: tl.constexpr = 4
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements
    r_indices = tl.arange(0, RBLOCK)[None, :]
    r_mask = r_indices < num_reduction_elements
    reduction_indices = r_indices
    input_indices = x_indices

    mean0 = tl.load(input_ptr_mean + (reduction_indices + 3 * input_indices), r_mask & x_mask, other=0.0)
    mean1 = tl.load(input_ptr_var + (reduction_indices + 3 * input_indices), r_mask & x_mask, other=0.0)
    mean2 = tl.load(input_ptr_count + (reduction_indices + 3 * input_indices), r_mask & x_mask, other=0.0)

    broadcast_mean0 = tl.broadcast_to(mean0, [XBLOCK, RBLOCK])
    broadcast_mean1 = tl.broadcast_to(mean1, [XBLOCK, RBLOCK])
    broadcast_mean2 = tl.broadcast_to(mean2, [XBLOCK, RBLOCK])

    masked_mean0 = tl.where(r_mask & x_mask, broadcast_mean0, 0)
    masked_mean1 = tl.where(r_mask & x_mask, broadcast_mean1, 0)
    masked_mean2 = tl.where(r_mask & x_mask, broadcast_mean2, 0)

    mean_accumulator, variance_accumulator, count_accumulator = triton_helpers.welford(masked_mean0, masked_mean1, masked_mean2, 1)

    mean_accumulator_expanded = mean_accumulator[:, None]
    variance_accumulator_expanded = variance_accumulator[:, None]

    kernel_size_squared = 8 * kernel_size * kernel_size
    kernel_size_squared_float = kernel_size_squared.to(tl.float32)

    normalized_variance = variance_accumulator_expanded / kernel_size_squared_float
    epsilon = 1e-05
    variance_with_epsilon = normalized_variance + epsilon

    reciprocal_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_with_epsilon)

    tl.store(output_ptr_count + (input_indices), reciprocal_sqrt_variance, x_mask)
    tl.store(output_ptr_mean + (input_indices), mean_accumulator_expanded, x_mask)
    tl.store(output_ptr_var + (input_indices), variance_accumulator_expanded, x_mask)