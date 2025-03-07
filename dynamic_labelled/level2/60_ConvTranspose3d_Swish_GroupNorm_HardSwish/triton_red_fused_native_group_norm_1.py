# From: 60_ConvTranspose3d_Swish_GroupNorm_HardSwish

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused_native_group_norm_1red_fused_native_group_norm_1(
    in_out_ptr, input_ptr, output_ptr, kernel_size_z, kernel_size_y, kernel_size_x, 
    num_elements_x, num_elements_r, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < num_elements_x
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_indices_flat = x_indices
    mean_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    m2_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    weight_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)

    for r_offset in range(0, num_elements_r, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < num_elements_r
        r_z = (r_indices % kernel_size_z)
        r_y = r_indices // kernel_size_z
        input_index = (
            (-1) * r_y + 
            (-1) * ((r_z // ((-1) + 2 * kernel_size_x)) % ((-1) + 2 * kernel_size_x)) + 
            (-4) * x_indices_flat + 
            (-16) * x_indices_flat * kernel_size_x * kernel_size_x + 
            (-4) * kernel_size_x * (triton_helpers.div_floor_integer(r_z, 1 + ((-4) * kernel_size_x) + 4 * kernel_size_x * kernel_size_x)) + 
            (-4) * r_y * kernel_size_x * kernel_size_x + 
            2 * kernel_size_y * r_y + 
            2 * kernel_size_x * ((r_z // ((-1) + 2 * kernel_size_x)) % ((-1) + 2 * kernel_size_x)) + 
            4 * kernel_size_x * r_y + 
            4 * kernel_size_x * kernel_size_x * (triton_helpers.div_floor_integer(r_z, 1 + ((-4) * kernel_size_x) + 4 * kernel_size_x * kernel_size_x)) + 
            8 * kernel_size_y * x_indices_flat + 
            16 * kernel_size_x * x_indices_flat + 
            (-32) * kernel_size_y * kernel_size_x * x_indices_flat + 
            (-8) * kernel_size_y * kernel_size_x * r_y + 
            8 * kernel_size_y * r_y * kernel_size_x * kernel_size_x + 
            32 * kernel_size_y * x_indices_flat * kernel_size_x * kernel_size_x + 
            (triton_helpers.div_floor_integer(r_z, 1 + ((-4) * kernel_size_x) + 4 * kernel_size_x * kernel_size_x)) + 
            (r_z % ((-1) + 2 * kernel_size_x))
        )
        input_values = tl.load(input_ptr + input_index, r_mask & x_mask, eviction_policy='evict_last', other=0.0)
        sigmoid_values = tl.sigmoid(input_values)
        product_values = sigmoid_values * input_values
        broadcasted_values = tl.broadcast_to(product_values, [XBLOCK, RBLOCK])
        mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
            broadcasted_values, mean_accumulator, m2_accumulator, weight_accumulator, r_offset == 0
        )
        mean_accumulator = tl.where(r_mask & x_mask, mean_next, mean_accumulator)
        m2_accumulator = tl.where(r_mask & x_mask, m2_next, m2_accumulator)
        weight_accumulator = tl.where(r_mask & x_mask, weight_next, weight_accumulator)

    mean_final, variance_final, weight_final = triton_helpers.welford(
        mean_accumulator, m2_accumulator, weight_accumulator, 1
    )
    mean_final_broadcast = mean_final[:, None]
    variance_final_broadcast = variance_final[:, None]
    weight_final_broadcast = weight_final[:, None]

    tl.store(output_ptr + (x_indices_flat), mean_final_broadcast, x_mask)
    epsilon = 1e-05
    variance_adjusted = variance_final_broadcast / (variance_final_broadcast + epsilon)
    inv_sqrt_variance = tl.extra.cuda.libdevice.rsqrt(variance_adjusted)
    tl.debug_barrier()
    tl.store(in_out_ptr + (x_indices_flat), inv_sqrt_variance, x_mask)