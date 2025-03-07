# From: 35_GroupNorm_

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_per_fused_native_group_norm_1(input_ptr_mean, input_ptr_var, input_ptr_count, output_ptr_mean, output_ptr_var, output_ptr_count, num_elements, num_groups, XBLOCK: tl.constexpr):
    num_elements = 128
    num_groups = 3
    GROUP_BLOCK: tl.constexpr = 4
    group_offset = tl.program_id(0) * XBLOCK
    element_index = group_offset + tl.arange(0, XBLOCK)[:, None]
    element_mask = element_index < num_elements
    group_index = tl.arange(0, GROUP_BLOCK)[None, :]
    group_mask = group_index < num_groups
    group_offset = group_index
    element_offset = element_index

    mean0 = tl.load(input_ptr_mean + (group_offset + (3 * element_offset)), group_mask & element_mask, other=0.0)
    mean1 = tl.load(input_ptr_var + (group_offset + (3 * element_offset)), group_mask & element_mask, other=0.0)
    mean2 = tl.load(input_ptr_count + (group_offset + (3 * element_offset)), group_mask & element_mask, other=0.0)

    broadcast_mean0 = tl.broadcast_to(mean0, [XBLOCK, GROUP_BLOCK])
    broadcast_mean1 = tl.broadcast_to(mean1, [XBLOCK, GROUP_BLOCK])
    broadcast_mean2 = tl.broadcast_to(mean2, [XBLOCK, GROUP_BLOCK])

    masked_mean0 = tl.where(group_mask & element_mask, broadcast_mean0, 0)
    masked_mean1 = tl.where(group_mask & element_mask, broadcast_mean1, 0)
    masked_mean2 = tl.where(group_mask & element_mask, broadcast_mean2, 0)

    mean, variance, count = triton_helpers.welford(masked_mean0, masked_mean1, masked_mean2, 1)

    reshaped_mean = mean[:, None]
    reshaped_variance = variance[:, None]

    normalization_factor = 524288.0
    epsilon = 1e-05

    adjusted_variance = reshaped_variance / normalization_factor
    variance_with_epsilon = adjusted_variance + epsilon
    reciprocal_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_with_epsilon)

    tl.store(output_ptr_count + (element_offset), reciprocal_sqrt_variance, element_mask)
    tl.store(output_ptr_mean + (element_offset), reshaped_mean, element_mask)
    tl.store(output_ptr_var + (element_offset), reshaped_variance, element_mask)