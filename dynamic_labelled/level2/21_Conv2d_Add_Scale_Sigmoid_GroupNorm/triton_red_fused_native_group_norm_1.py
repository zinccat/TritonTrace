# From: 21_Conv2d_Add_Scale_Sigmoid_GroupNorm

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_group_norm_1red_fused_native_group_norm_1(
    input_ptr_mean, input_ptr_var, input_ptr_weight, 
    output_ptr_mean, output_ptr_var, output_ptr_inv_std, 
    kernel_size_0, kernel_size_1, num_elements_x, num_elements_r, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < num_elements_x
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_channel = x_index
    x_within_block = (x_index % 8)
    running_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)

    for r_offset in range(0, num_elements_r, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < num_elements_r
        r_within_kernel_0 = (r_index % kernel_size_0)
        r_within_kernel_1 = r_index // kernel_size_0

        input_index_mean = (
            (-2) * (triton_helpers.div_floor_integer(r_within_kernel_0, (-2) + kernel_size_1)) 
            + 4 * r_within_kernel_1 
            + 8 * x_channel 
            + kernel_size_1 * (triton_helpers.div_floor_integer(r_within_kernel_0, (-2) + kernel_size_1)) 
            + r_within_kernel_1 * kernel_size_1 * kernel_size_1 
            + (-8) * kernel_size_1 * x_channel 
            + (-4) * kernel_size_1 * r_within_kernel_1 
            + 2 * x_channel * kernel_size_1 * kernel_size_1 
            + (r_within_kernel_0 % (-2 + kernel_size_1))
        )

        input_index_var = r_within_kernel_1 + 2 * x_within_block
        input_index_weight = r_within_kernel_1 + 2 * x_within_block

        input_mean = tl.load(input_ptr_mean + input_index_mean, r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        input_var = tl.load(input_ptr_var + input_index_var, r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        input_weight = tl.load(input_ptr_weight + input_index_weight, r_mask & x_mask, eviction_policy='evict_last', other=0.0)

        weighted_input = input_mean + input_var
        weighted_input_scaled = weighted_input * input_weight
        sigmoid_output = tl.sigmoid(weighted_input_scaled)
        broadcast_sigmoid = tl.broadcast_to(sigmoid_output, [XBLOCK, RBLOCK])

        running_mean_next, running_m2_next, running_weight_next = triton_helpers.welford_reduce(
            broadcast_sigmoid, running_mean, running_m2, running_weight, r_offset == 0
        )

        running_mean = tl.where(r_mask & x_mask, running_mean_next, running_mean)
        running_m2 = tl.where(r_mask & x_mask, running_m2_next, running_m2)
        running_weight = tl.where(r_mask & x_mask, running_weight_next, running_weight)

    final_mean, final_m2, final_weight = triton_helpers.welford(
        running_mean, running_m2, running_weight, 1
    )

    final_mean_broadcast = final_mean[:, None]
    final_m2_broadcast = final_m2[:, None]

    tl.store(output_ptr_mean + x_channel, final_mean_broadcast, x_mask)
    tl.store(output_ptr_var + x_channel, final_m2_broadcast, x_mask)

    epsilon = 1e-05
    variance_adjustment = (
        (tl.full([], 0.0, tl.float64) * (tl.full([], 0.0, tl.float64) >= (8 + (-8) * kernel_size_1 + 2 * kernel_size_1 * kernel_size_1))
        + (8 + (-8) * kernel_size_1 + 2 * kernel_size_1 * kernel_size_1) * ((8 + (-8) * kernel_size_1 + 2 * kernel_size_1 * kernel_size_1) > tl.full([], 0.0, tl.float64))
    )
    variance_adjustment = variance_adjustment.to(tl.float32)

    inv_std = (final_m2_broadcast / variance_adjustment + epsilon).rsqrt()
    tl.store(output_ptr_inv_std + x_channel, inv_std, x_mask)