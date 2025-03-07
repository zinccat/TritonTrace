# From: 83_Conv3d_GroupNorm_Min_Clamp_Dropout

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_group_norm_2red_fused_native_group_norm_2(
    input_output_ptr, input_ptr, output_ptr, kernel_size_0, kernel_size_1, 
    x_num_elements, r_num_elements, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < x_num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_indices = x_index
    mean_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    m2_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    weight_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)

    for r_offset in range(0, r_num_elements, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < r_num_elements
        r_indices = r_index
        loaded_data = tl.load(
            input_ptr + (r_indices + ((-16) * x_indices) + ((-4) * x_indices * kernel_size_1 * kernel_size_1) + 8 * kernel_size_0 * x_indices + 16 * kernel_size_1 * x_indices + ((-8) * kernel_size_0 * kernel_size_1 * x_indices) + 2 * kernel_size_0 * x_indices * kernel_size_1 * kernel_size_1),
            r_mask & x_mask,
            eviction_policy='evict_first',
            other=0.0
        )
        broadcast_data = tl.broadcast_to(loaded_data, [XBLOCK, RBLOCK])
        mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
            broadcast_data, mean_accumulator, m2_accumulator, weight_accumulator, r_offset == 0
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

    tl.store(output_ptr + (x_indices), mean_result, x_mask)

    min_value = tl.full([], 0.0, tl.float64)
    max_value = tl.full([], 0.0, tl.float64)
    clamp_value = ((-16) + ((-4) * kernel_size_1 * kernel_size_1) + 8 * kernel_size_0 + 16 * kernel_size_1 + ((-8) * kernel_size_0 * kernel_size_1) + 2 * kernel_size_0 * kernel_size_1 * kernel_size_1)
    clamped_value = tl.where(min_value >= clamp_value, min_value, tl.where(clamp_value > max_value, clamp_value, clamp_value))
    clamped_value = clamped_value.to(tl.float32)

    variance_adjusted = variance_result / clamped_value
    epsilon = 1e-05
    variance_adjusted += epsilon
    rsqrt_result = tl.extra.cuda.libdevice.rsqrt(variance_adjusted)

    tl.debug_barrier()
    tl.store(input_output_ptr + (x_indices), rsqrt_result, x_mask)