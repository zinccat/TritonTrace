# From: 46_NetVladWithGhostClusters

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_1red_fused__native_batch_norm_legit_functional_1(
    input_ptr, output_ptr_mean, output_ptr_var, output_ptr_weight, 
    total_elements, reduction_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    total_elements = 1200
    reduction_elements = 128
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < total_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_channel = (x_indices % 48)
    x_sample = x_indices // 48
    running_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x_flat_index = x_indices

    for r_offset in range(0, reduction_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < reduction_elements
        r_channel = r_indices
        input_data = tl.load(
            input_ptr + (x_channel + 48 * r_channel + 6144 * x_sample), 
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

    mean, variance, weight = triton_helpers.welford(
        running_mean, running_m2, running_weight, 1
    )
    mean = mean[:, None]
    variance = variance[:, None]
    weight = weight[:, None]

    tl.store(output_ptr_mean + (x_flat_index), mean, x_mask)
    tl.store(output_ptr_var + (x_flat_index), variance, x_mask)
    tl.store(output_ptr_weight + (x_flat_index), weight, x_mask)