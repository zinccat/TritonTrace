# From: 34_InstanceNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_0(
    input_ptr, output_ptr, kernel_size, num_elements_x, num_elements_r, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
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
        input_values = tl.load(
            input_ptr + (r_indices_flat + x_indices_flat * kernel_size * kernel_size), 
            r_mask & x_mask, 
            eviction_policy='evict_last', 
            other=0.0
        )
        broadcasted_values = tl.broadcast_to(input_values, [XBLOCK, RBLOCK])
        running_mean_next, running_m2_next, running_weight_next = triton_helpers.welford_reduce(
            broadcasted_values, running_mean, running_m2, running_weight, r_offset == 0
        )
        running_mean = tl.where(r_mask & x_mask, running_mean_next, running_mean)
        running_m2 = tl.where(r_mask & x_mask, running_m2_next, running_m2)
        running_weight = tl.where(r_mask & x_mask, running_weight_next, running_weight)

    mean, variance, weight = triton_helpers.welford(running_mean, running_m2, running_weight, 1)
    mean_broadcasted = mean[:, None]
    variance_broadcasted = variance[:, None]
    weight_broadcasted = weight[:, None]

    for r_offset in range(0, num_elements_r, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < num_elements_r
        r_indices_flat = r_indices
        input_values = tl.load(
            input_ptr + (r_indices_flat + x_indices_flat * kernel_size * kernel_size), 
            r_mask & x_mask, 
            eviction_policy='evict_first', 
            other=0.0
        )
        centered_values = input_values - mean_broadcasted
        kernel_area = kernel_size * kernel_size
        kernel_area_float = kernel_area.to(tl.float32)
        variance_normalized = variance_broadcasted / kernel_area_float
        epsilon = 1e-05
        variance_stabilized = variance_normalized + epsilon
        inv_stddev = tl.extra.cuda.libdevice.rsqrt(variance_stabilized)
        normalized_values = centered_values * inv_stddev
        tl.store(output_ptr + (r_indices_flat + x_indices_flat * kernel_size * kernel_size), normalized_values, r_mask & x_mask)