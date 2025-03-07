# From: 41_Gemm_BatchNorm_GELU_GroupNorm_Mean_ReLU

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_0(
    input_ptr_mean, input_ptr_var, input_ptr_bias, 
    output_ptr_normalized, output_ptr_scale, output_ptr_bias, 
    kernel_size, total_elements, reduction_elements, 
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr
):
    total_elements = 1024
    x_offset = tl.program_id(0) * XBLOCK
    x_indices = x_offset + tl.arange(0, XBLOCK)[:, None]
    x_mask = x_indices < total_elements
    r_base = tl.arange(0, RBLOCK)[None, :]
    x_indices_flat = x_indices
    running_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    running_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)

    for r_offset in range(0, reduction_elements, RBLOCK):
        r_indices = r_offset + r_base
        r_mask = r_indices < reduction_elements
        r_indices_flat = r_indices
        input_data = tl.load(
            input_ptr_mean + (x_indices_flat + 1024 * r_indices_flat), 
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
    mean_broadcast = mean[:, None]
    variance_broadcast = variance[:, None]
    weight_broadcast = weight[:, None]

    tl.store(output_ptr_normalized + (x_indices_flat), mean_broadcast, x_mask)
    input_scale = tl.load(input_ptr_var + (x_indices_flat), x_mask, eviction_policy='evict_last')
    input_bias = tl.load(input_ptr_bias + (x_indices_flat), x_mask, eviction_policy='evict_last')
    kernel_size_float = kernel_size.to(tl.float32)
    inv_stddev = variance_broadcast / kernel_size_float
    epsilon = 1e-05
    inv_stddev_adjusted = inv_stddev + epsilon
    inv_stddev_final = tl.extra.cuda.libdevice.rsqrt(inv_stddev_adjusted)
    mean_adjustment = (1024 * kernel_size) / 1024 / ((tl.full([], -1.0, tl.float64)) + (1024 * kernel_size) / 1024)
    mean_adjustment_float = mean_adjustment.to(tl.float32)
    mean_scaled = variance_broadcast * mean_adjustment_float
    momentum = 0.1
    mean_scaled_momentum = mean_scaled * momentum
    scale_factor = 0.9
    scaled_input_scale = input_scale * scale_factor
    updated_scale = mean_scaled_momentum + scaled_input_scale
    normalized_data = mean_broadcast * momentum
    scaled_input_bias = input_bias * scale_factor
    updated_bias = normalized_data + scaled_input_bias

    tl.store(output_ptr_scale + (x_indices_flat), inv_stddev_final, x_mask)
    tl.store(output_ptr_bias + (x_indices_flat), updated_scale, x_mask)
    tl.store(output_ptr_bias + (x_indices_flat), updated_bias, x_mask)