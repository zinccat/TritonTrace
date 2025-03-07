# From: 79_Conv3d_Multiply_InstanceNorm_Clamp_Multiply_Max

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_1red_fused__native_batch_norm_legit_1(
    in_out_ptr, input_ptr, scale_ptr, output_ptr, kernel_size0, kernel_size1, x_num_elements, r_num_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    x_offset = tl.program_id(0) * XBLOCK
    x_index = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_index < x_num_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x0 = x_index
    scale_values = tl.load(scale_ptr + ((x0 % 16)), x_mask, eviction_policy='evict_last')
    mean_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    m2_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    weight_accumulator = tl.zeros([XBLOCK, RBLOCK], tl.float32)

    for r_offset in range(0, r_num_elements, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < r_num_elements
        r1 = r_index
        input_values = tl.load(
            input_ptr + (r1 + ((-8) * x0) + ((-2) * x0 * kernel_size1 * kernel_size1) + 4 * kernel_size0 * x0 + 8 * kernel_size1 * x0 + kernel_size0 * x0 * kernel_size1 * kernel_size1 + ((-4) * kernel_size0 * kernel_size1 * x0)),
            rmask & x_mask,
            eviction_policy='evict_first',
            other=0.0
        )
        scaled_values = input_values * scale_values
        broadcasted_values = tl.broadcast_to(scaled_values, [XBLOCK, RBLOCK])
        mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
            broadcasted_values, mean_accumulator, m2_accumulator, weight_accumulator, r_offset == 0
        )
        mean_accumulator = tl.where(rmask & x_mask, mean_next, mean_accumulator)
        m2_accumulator = tl.where(rmask & x_mask, m2_next, m2_accumulator)
        weight_accumulator = tl.where(rmask & x_mask, weight_next, weight_accumulator)

    mean_final, variance_final, weight_final = triton_helpers.welford(
        mean_accumulator, m2_accumulator, weight_accumulator, 1
    )
    mean_final = mean_final[:, None]
    variance_final = variance_final[:, None]
    weight_final = weight_final[:, None]

    tl.store(output_ptr + (x0), mean_final, x_mask)
    clamp_value = ((tl.full([], 0.0, tl.float64)) * ((tl.full([], 0.0, tl.float64)) >= ((-8) + ((-2) * kernel_size1 * kernel_size1) + 4 * kernel_size0 + 8 * kernel_size1 + kernel_size0 * kernel_size1 * kernel_size1 + ((-4) * kernel_size0 * kernel_size1))) + ((-8) + ((-2) * kernel_size1 * kernel_size1) + 4 * kernel_size0 + 8 * kernel_size1 + kernel_size0 * kernel_size1 * kernel_size1 + ((-4) * kernel_size0 * kernel_size1)) * (((-8) + ((-2) * kernel_size1 * kernel_size1) + 4 * kernel_size0 + 8 * kernel_size1 + kernel_size0 * kernel_size1 * kernel_size1 + ((-4) * kernel_size0 * kernel_size1)) > (tl.full([], 0.0, tl.float64))))
    clamp_value = clamp_value.to(tl.float32)
    normalized_variance = variance_final / clamp_value
    epsilon = 1e-05
    adjusted_variance = normalized_variance + epsilon
    reciprocal_sqrt = tl.extra.cuda.libdevice.rsqrt(adjusted_variance)
    tl.debug_barrier()
    tl.store(in_out_ptr + (x0), reciprocal_sqrt, x_mask)